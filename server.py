import asyncio
import logging
from typing import Dict, Any
import httpx
from mcp.server.fastmcp import FastMCP
import json

# Minimal logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

import os
from dotenv import load_dotenv


mcp = FastMCP("Travel_server")
API_BASE_URL = os.environ.get('API_BASE_URL')

async def clean_textcontent_list(raw_list: list) -> list[Dict]:
    """
    Transforme une liste de TextContent en vraie liste de dictionnaires.
    """
    cleaned = []
    for item in raw_list:
        # item.text contient le JSON sous forme de string
        try:
            data = json.loads(item.text)
            cleaned.append(data)
        except Exception as e:
            print(f"Erreur lors du parsing JSON: {e}")
            continue
    return cleaned

async def simple_request(url: str) -> list[Dict]:
    """Simple HTTP request with minimal retry"""
    headers = {'Content-Type': 'application/json', 'User-Agent': 'EaseTravel-Assistant'}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=15.0)
            response.raise_for_status()
            data = await response.json()
            return data if isinstance(data, list) else [data]
    except Exception as e:
        print(f"error {e}")
        # Single retry on failure
        try:
            await asyncio.sleep(0.5)
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=15.0)
                response.raise_for_status()
                data = response.json()
                print(data)
                return data if isinstance(data, list) else [data]
        except Exception:
            return []

@mcp.tool()
async def get_airport_iata_code(city: str) -> list[Dict]:
    """
    Get airport information for a specified city.
    
    This tool searches for airports in or near the specified city and returns
    airport codes (IATA), names, and location information.
    
    Args:
        city: Name of the city to search for airports (e.g., "Paris", "New York")
    
    Returns:
        Dictionary containing airport information including IATA codes, names, and locations 
    """
    if not city or len(city.strip()) < 2:
        return {"success": False, "error": "City name too short"}
    
    url = f"{API_BASE_URL}/place?text={city.strip()}"
    result = await simple_request(url)

    data = result

    return data
    

    

@mcp.tool() 
async def one_way_flights(dep_airport_code: str, arr_airport_code: str, start_date: str, 
                        adults: int = 1, children: int = 0, infants: int = 0) -> list[Dict]:
    """
    Search for flights between airports
    
    Args:
        departure_airport: Departure airport code (e.g., "CDG-AIRPORT")
        arrival_airport: Arrival airport code (e.g., "NSI-AIRPORT")
        start_date: Departure date in format "DD-MM-YYYY"
        adults: Number of adults (default: 1)
        children: Number of children (default: 0)
        infants: Number of infants (default: 0)
        
    Returns:
        Dictionary containing flight search results or None if error
    """
    
    # Quick validation
    if not dep_airport_code or not arr_airport_code:
        return {"success": False, "error": "Airport codes required"}
    
    if dep_airport_code.strip().upper() == arr_airport_code.strip().upper():
        return {"success": False, "error": "Same departure and arrival airport"}
    
    if adults < 1 or adults > 9:
        return {"success": False, "error": "Adults must be 1-9"}
    
    # Clean codes
    dep = dep_airport_code.strip().upper()
    arr = arr_airport_code.strip().upper()
    
    if not dep.endswith('-AIRPORT'):
        dep += '-AIRPORT'
    if not arr.endswith('-AIRPORT'):
        arr += '-AIRPORT'
    
    url = f"{API_BASE_URL}/search?departure={dep.strip()}&arrival={arr.strip()}&adult={adults}&children={children}&infant={infants}&startDate={start_date.strip()}&locale=fr-cm"
    
    result = await simple_request(url)

    return result


@mcp.tool() 
async def round_trip_flights(dep_airport_code: str, arr_airport_code: str, start_date: str, return_date: str,
                        adults: int = 1, children: int = 0, infants: int = 0) -> list[Dict]:
    """
    Search for flights between airports
    
    Args:
        departure_airport: Departure airport code (e.g., "CDG-AIRPORT")
        arrival_airport: Arrival airport code (e.g., "NSI-AIRPORT")
        start_date: Departure date in format "DD-MM-YYYY"
        adults: Number of adults (default: 1)
        children: Number of children (default: 0)
        infants: Number of infants (default: 0)
        
    Returns:
        Dictionary containing flight search results or None if error
    """
    
    # Quick validation
    if not dep_airport_code or not arr_airport_code:
        return {"success": False, "error": "Airport codes required"}
    
    if dep_airport_code.strip().upper() == arr_airport_code.strip().upper():
        return {"success": False, "error": "Same departure and arrival airport"}
    
    if adults < 1 or adults > 9:
        return {"success": False, "error": "Adults must be 1-9"}
    
    # Clean codes
    dep = dep_airport_code.strip().upper()
    arr = arr_airport_code.strip().upper()
    
    if not dep.endswith('-AIRPORT'):
        dep += '-AIRPORT'
    if not arr.endswith('-AIRPORT'):
        arr += '-AIRPORT'
    
    url = f"{API_BASE_URL}/search?departure={dep.strip()}&arrival={arr.strip()}&startDate={start_date.strip()}&endDate={return_date.strip()}&adult={adults}&children={children}&infant={infants}&locale=fr-cm"
    
    result = await simple_request(url)

    return result
  
  
    

if __name__ == "__main__":
    print("🚀 Fast Travel Server")
    mcp.run(transport='stdio')