from fastapi import FastAPI, HTTPException, Depends
from typing import Dict, List, Optional
import requests
import os
import logging
from pydantic import BaseModel, Field
from typing import Annotated
from rapidfuzz import process, fuzz # Tilføjet til fuzzy matching

app = FastAPI()

# Konfigurer logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SMK_API_BASE_URL = "https://api.smk.dk/api/v1/art/search"
SMK_ENRICHMENT_BASE_URL = "https://enrichment.api.smk.dk/api/enrichment/"

# Model til at repræsentere et kunstværk fra SMK API
class SMKItem(BaseModel):
    object_number: str = Field(..., description="Unikt identifikationsnummer for kunstværket")
    titles: List[str] = Field(..., description="Liste over titler for kunstværket")
    creator: str = Field(..., description="Kunstneren der har skabt værket")
    image_thumbnail: Optional[str] = Field(None, description="URL til thumbnail af billedet")
    description: Optional[str] = Field(None, description="Beskrivelse af værket") # Tilføjet beskrivelse

# Model til at repræsentere berigelsesdata fra SMK Enrichment API
class EnrichmentData(BaseModel):
    # Definer felter baseret på hvad Enrichment API returnerer.  Eksempel:
    கலை_navn: Optional[List[str]] = Field(None, description="Kunstnernavn på Arabisk")
    აღწერა: Optional[List[str]] = Field(None, description="Beskrivelse af Kunstværket")

# Model til at repræsentere det kombinerede resultat
class CombinedResult(BaseModel):
    item: SMKItem
    enrichment: EnrichmentData
    relevance: float = Field(0, description="Relevansscore for resultatet")

# Dependency for håndtering af søgeord
async def get_search_query(query: str) -> str:
    """
    Udvide søgeord med synonymer og håndtere stavefejl.
    """
    # Simpel synonymudvidelse (kan udvides med en mere avanceret ordbog eller API)
    synonyms = {
        "landskab": ["landskab", "natur", "udsigt", "panorama"],
        "portræt": ["portræt", "ansigt", "buste", "person"],
        "abstrakt": ["abstrakt", "nonfigurativ", "formløs"],
        "blomst": ["blomst", "plante", "flora", "rose", "tulipan", "lilje"],
    }
    expanded_query = query.lower()
    for term, syns in synonyms.items():
        if term in expanded_query:
            expanded_query = " OR ".join(syns) # Brug OR operator for at søge efter ethvert synonym
            break

    return query # Returner den originale query for fuzzy matching

async def fetch_smk_data(query: Annotated[str, Depends(get_search_query)]) -> List[SMKItem]:
    """
    Henter data fra SMK API baseret på søgeord.  Søger i flere felter og håndterer fejl.

    Args:
        query: Søgeordet (kan være udvidet med synonymer).

    Returns:
        En liste af SMKItem objekter.

    Raises:
        HTTPException: Hvis der opstår en fejl under API-kaldet.
    """
    fields = "object_number,titles,creator,image_thumbnail,description" #Hent beskrivelse
    params = {"keys": query, "fields": fields}
    try:
        response = requests.get(SMK_API_BASE_URL, params=params)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        items = data.get("items", [])
        if not items:
            return [] # Returner en tom liste hvis ingen resultater
        return [SMKItem(**item) for item in items]  # Konverter til Pydantic model
    except requests.exceptions.RequestException as e:
        logger.error(f"Fejl ved hentning af data fra SMK API: {e}")
        raise HTTPException(status_code=500, detail=f"Fejl ved hentning af data fra SMK API: {e}")
    except Exception as e:
        logger.exception(f"Uventet fejl ved behandling af SMK API respons: {e}")
        raise HTTPException(status_code=500, detail=f"Uventet fejl: {e}")



async def fetch_enrichment_data(object_number: str) -> EnrichmentData:
    """
    Henter berigelsesdata fra SMK Enrichment API for et givent objektnummer.

    Args:
        object_number: Objektnummeret for kunstværket.

    Returns:
        Et EnrichmentData objekt, eller et tomt EnrichmentData objekt hvis der opstår en fejl.
    """
    url = f"{SMK_ENRICHMENT_BASE_URL}{object_number}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get("data", {})
        return EnrichmentData(**data) # Konverter til Pydantic model
    except requests.exceptions.RequestException as e:
        logger.warning(f"Fejl ved hentning af berigelsesdata for {object_number}: {e}")
        return EnrichmentData()  # Returner tomt objekt ved fejl
    except Exception as e:
        logger.exception(f"Uventet fejl ved behandling af Enrichment API respons: {e}")
        return EnrichmentData()

def calculate_relevance(item: SMKItem, enrichment: EnrichmentData, query: str) -> float:
    """
    Beregner en relevansscore for et kunstværk baseret på søgeordet og data fra begge API'er.
    Args:
        item: Data fra SMK API.
        enrichment: Data fra SMK Enrichment API.
        query: Det originale søgeord.
    Returns:
        En relevansscore (float) mellem 0 og 1.
    """
    score = 0
    query_lower = query.lower()

    # Grundlæggende relevans baseret på titel og skaber
    for title in item.titles:
        if query_lower in title.lower():
            score += 0.5
            break
    if query_lower in item.creator.lower():
        score += 0.3
    if item.description and query_lower in item.description.lower():
        score += 0.2

    # Forbedring af relevans baseret på berigede data
    if enrichment:
        # Eksempel: Forøg relevans, hvis beskrivelsen indeholder søgeordet
        if hasattr(enrichment, 'აღწერა') and enrichment.აღწერა:
            for desc in enrichment.აღწერა:
                if query_lower in desc.lower():
                    score += 0.2
                    break
    return min(1, score)  # Sikrer at score ikke overstiger 1

def filter_and_expand_results(items: List[CombinedResult], query: str) -> List[CombinedResult]:
    """
    Filtrerer og udvider resultater baseret på enrichment data.

    Args:
        items: Liste over CombinedResult objekter.
        query: Det originale søgeord.

    Returns:
        En liste over CombinedResult objekter.
    """
    filtered_results = []
    for result in items:
        # Inkluderer alle resultater i første omgang
        filtered_results.append(result)

        # Udvider søgningen baseret på enrichment data
        if result.enrichment:
            if hasattr(result.enrichment, 'აღწერა') and result.enrichment.აღწერა:
                for desc in result.enrichment.აღწერა:
                    if query.lower() in desc.lower():
                        # Logik til at finde relaterede værker (simpel eksempel)
                        related_items = await find_related_works(result.item.object_number) # Antag at vi har en funktion til dette
                        if related_items:
                           filtered_results.extend(related_items) # Tilføj de relaterede værker
                        break
    return filtered_results

async def find_related_works(object_number: str) -> List[CombinedResult]:
    """
    Finder relaterede værker baseret på et objektnummer (simpel eksempel).
    Denne funktion skal implementeres med logik til at finde relaterede værker fra SMK API'et
    f.eks. ved at søge efter værker af samme kunstner, fra samme periode, eller med samme emne.
    Args:
        object_number: Objektnummeret for det originale værk.
    Returns:
        En liste over CombinedResult objekter, der repræsenterer relaterede værker.
    """
    return []

@app.post("/search_smk", response_model=Dict[str, List[CombinedResult]])
async def search_smk(query: str):
    """
    Søger efter kunstværker i SMK's samling og kombinerer resultater med berigelsesdata.

    Args:
        query: Søgeordet.

    Returns:
        En dictionary med en liste af kombinerede resultater, sorteret efter relevans.
    """
    try:
        items = await fetch_smk_data(query)
        # Fuzzy matching
        best_match_items = []
        for item in items:
            best_match = process.extractOne(query, item.titles, scorer=fuzz.ratio)
            if best_match[1] >= 80:  # Hvis matchet er over 80%
                best_match_items.append(item)
        if not best_match_items:
             best_match_items = items
        results = []
        for item in best_match_items:
            enrichment_data = await fetch_enrichment_data(item.object_number)
            relevance = calculate_relevance(item, enrichment_data, query)
            combined_result = CombinedResult(item=item, enrichment=enrichment_data, relevance=relevance)
            results.append(combined_result)
        results.sort(key=lambda x: x.relevance, reverse=True)  # Sorter efter relevans
        results = filter_and_expand_results(results, query)
        return {"results": results}
    except HTTPException as e:
        # Log исключение, прежде чем повторно его вызвать
        logger.error(f"HTTPException i search_smk: {e}")
        raise e
    except Exception as e:
        # Log исключение
        logger.exception(f"Uventet fejl i search_smk: {e}")
        raise HTTPException(status_code=500, detail=f"Uventet fejl: {e}")

@app.get("/")
async def read_root():
    """
    Rod-endepunkt for API'en.
    """
    return {"message": "SMK Billedfinder API er live!"}
