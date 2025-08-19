"""
IPC Classification API Client
Module for interacting with WIPO IPC classification service
"""

import xml.etree.ElementTree as ET
import requests
from typing import List, Dict, Any


def format_ipc_code(raw_code: str) -> str:
    """
    Format raw IPC code into standard format.
    
    Args:
        raw_code: Raw IPC code string
        
    Returns:
        Formatted IPC code
    """
    section = raw_code[0]
    class_ = raw_code[1:3]
    subclass = raw_code[3]
    main_group = raw_code[4:8].lstrip('0')
    subgroup = raw_code[8:10] + raw_code[10:].rstrip('0')
    return f"{section}{class_}{subclass}{main_group}/{subgroup}"


def parse_predictions(xml_string: str) -> List[Dict[str, Any]]:
    """
    Parse XML response from IPC classification API.
    
    Args:
        xml_string: XML response string
        
    Returns:
        List of prediction dictionaries with rank, category, and score
    """
    root = ET.fromstring(xml_string)
    predictions = []
    
    for pred in root.findall('prediction'):
        rank = pred.find('rank').text if pred.find('rank') is not None else None
        category = pred.find('category').text if pred.find('category') is not None else None
        score = pred.find('score').text if pred.find('score') is not None else None
        
        predictions.append({
            "rank": int(rank) if rank is not None else None,
            "category": format_ipc_code(category),
            "score": int(score) if score is not None else None
        })
    
    return predictions


def get_ipc_classification(query: str) -> str:
    """
    Get IPC classification for a given query text.
    
    Args:
        query: Text to classify
        
    Returns:
        XML response string
    """
    xml_data = f"""<?xml version="1.0" encoding="UTF-8"?>
<request>
  <lang>en</lang>
  <text>{query}</text>
  <numberofpredictions>3</numberofpredictions>
  <hierarchiclevel>SUBGROUP</hierarchiclevel>
</request>"""
    
    headers = {
        'Content-Type': 'application/xml'
    }
    
    response = requests.post(
        url='https://ipccat.wipo.int/EN/query',
        data=xml_data,
        headers=headers
    )
    
    print(f"IPC API Status: {response.status_code}")
    return response.text


def get_ipc_predictions(query: str) -> List[Dict[str, Any]]:
    """
    Get IPC classification predictions for a query.
    
    Args:
        query: Text to classify
        
    Returns:
        List of classification predictions
    """
    xml_response = get_ipc_classification(query)
    predictions = parse_predictions(xml_response)
    return predictions
