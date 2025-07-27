"""
XMP sidecar file support utilities.

Handles reading and writing metadata to XMP sidecar files for non-destructive
metadata editing that's compatible with other photo applications.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom

logger = logging.getLogger(__name__)

# XMP namespaces
XMP_NAMESPACES = {
    'x': 'adobe:ns:meta/',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'xmp': 'http://ns.adobe.com/xap/1.0/',
    'xmpRights': 'http://ns.adobe.com/xap/1.0/rights/',
    'photoshop': 'http://ns.adobe.com/photoshop/1.0/',
    'Iptc4xmpCore': 'http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/',
    'exif': 'http://ns.adobe.com/exif/1.0/',
    'tiff': 'http://ns.adobe.com/tiff/1.0/',
}


class XMPSidecar:
    """Handles XMP sidecar file operations."""
    
    def __init__(self, image_path: str):
        """Initialize with the path to the image file."""
        self.image_path = image_path
        self.sidecar_path = self._get_sidecar_path()
        
    def _get_sidecar_path(self) -> str:
        """Get the path for the XMP sidecar file."""
        base_path = os.path.splitext(self.image_path)[0]
        return f"{base_path}.xmp"
    
    def exists(self) -> bool:
        """Check if XMP sidecar file exists."""
        return os.path.exists(self.sidecar_path)
    
    def read(self) -> Dict[str, Any]:
        """Read metadata from XMP sidecar file."""
        if not self.exists():
            return {}
        
        try:
            tree = ET.parse(self.sidecar_path)
            root = tree.getroot()
            
            # Register namespaces
            for prefix, uri in XMP_NAMESPACES.items():
                ET.register_namespace(prefix, uri)
            
            metadata = {}
            
            # Extract keywords
            keywords = self._extract_keywords(root)
            if keywords:
                metadata['keywords'] = keywords
            
            # Extract IPTC metadata
            iptc_data = self._extract_iptc(root)
            if iptc_data:
                metadata['iptc'] = iptc_data
            
            # Extract Dublin Core metadata
            dc_data = self._extract_dublin_core(root)
            if dc_data:
                metadata['dc'] = dc_data
            
            # Extract rights metadata
            rights_data = self._extract_rights(root)
            if rights_data:
                metadata['rights'] = rights_data
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to read XMP sidecar {self.sidecar_path}: {e}")
            return {}
    
    def write(self, metadata: Dict[str, Any]) -> bool:
        """Write metadata to XMP sidecar file."""
        try:
            # Create or load existing XMP structure
            if self.exists():
                tree = ET.parse(self.sidecar_path)
                root = tree.getroot()
            else:
                root = self._create_xmp_structure()
                tree = ET.ElementTree(root)
            
            # Register namespaces
            for prefix, uri in XMP_NAMESPACES.items():
                ET.register_namespace(prefix, uri)
            
            # Find or create RDF Description
            rdf_root = root.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF')
            if rdf_root is None:
                rdf_root = ET.SubElement(root, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF')
            
            description = rdf_root.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description')
            if description is None:
                description = ET.SubElement(rdf_root, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description')
                description.set('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about', '')
            
            # Write keywords
            if 'keywords' in metadata:
                self._write_keywords(description, metadata['keywords'])
            
            # Write IPTC metadata
            if 'iptc' in metadata:
                self._write_iptc(description, metadata['iptc'])
            
            # Write Dublin Core metadata
            if 'dc' in metadata:
                self._write_dublin_core(description, metadata['dc'])
            
            # Write rights metadata
            if 'rights' in metadata:
                self._write_rights(description, metadata['rights'])
            
            # Pretty print and save
            xml_str = self._prettify_xml(root)
            with open(self.sidecar_path, 'w', encoding='utf-8') as f:
                f.write(xml_str)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write XMP sidecar {self.sidecar_path}: {e}")
            return False
    
    def sync_with_db(self, db_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sync XMP sidecar with database metadata."""
        xmp_metadata = self.read()
        
        # Merge metadata, preferring database values for conflicts
        merged = {
            'keywords': list(set(
                db_metadata.get('keywords', []) + 
                xmp_metadata.get('keywords', [])
            )),
            'iptc': {**xmp_metadata.get('iptc', {}), **db_metadata.get('iptc', {})},
            'dc': {**xmp_metadata.get('dc', {}), **db_metadata.get('dc', {})},
            'rights': {**xmp_metadata.get('rights', {}), **db_metadata.get('rights', {})}
        }
        
        # Write merged metadata back to XMP
        self.write(merged)
        
        return merged
    
    def _create_xmp_structure(self) -> ET.Element:
        """Create basic XMP structure."""
        # Create XMP packet
        xmp_root = ET.Element('x:xmpmeta')
        xmp_root.set('xmlns:x', XMP_NAMESPACES['x'])
        
        # Add RDF root
        rdf_root = ET.SubElement(xmp_root, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF')
        
        # Add Description
        description = ET.SubElement(rdf_root, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description')
        description.set('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about', '')
        
        # Set namespaces
        for prefix, uri in XMP_NAMESPACES.items():
            if prefix != 'x' and prefix != 'rdf':
                description.set(f'xmlns:{prefix}', uri)
        
        return xmp_root
    
    def _extract_keywords(self, root: ET.Element) -> List[str]:
        """Extract keywords from XMP."""
        keywords = []
        
        # Look for dc:subject
        subject_elem = root.find('.//{http://purl.org/dc/elements/1.1/}subject')
        if subject_elem is not None:
            bag_elem = subject_elem.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Bag')
            if bag_elem is not None:
                for li in bag_elem.findall('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li'):
                    if li.text:
                        keywords.append(li.text)
        
        return keywords
    
    def _extract_iptc(self, root: ET.Element) -> Dict[str, Any]:
        """Extract IPTC metadata from XMP."""
        iptc_data = {}
        
        description = root.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description')
        if description is None:
            return iptc_data
        
        # Map of IPTC fields to extract
        iptc_fields = {
            '{http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/}CreatorContactInfo': 'creator_contact',
            '{http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/}Location': 'sublocation',
            '{http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/}City': 'city',
            '{http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/}State': 'region',
            '{http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/}Country': 'country',
            '{http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/}CountryCode': 'country_code',
            '{http://ns.adobe.com/photoshop/1.0/}Headline': 'headline',
            '{http://ns.adobe.com/photoshop/1.0/}Instructions': 'instructions',
            '{http://ns.adobe.com/photoshop/1.0/}Credit': 'credit',
            '{http://ns.adobe.com/photoshop/1.0/}Source': 'source',
        }
        
        for xmp_field, iptc_field in iptc_fields.items():
            elem = description.find(xmp_field)
            if elem is not None and elem.text:
                iptc_data[iptc_field] = elem.text
        
        return iptc_data
    
    def _extract_dublin_core(self, root: ET.Element) -> Dict[str, Any]:
        """Extract Dublin Core metadata from XMP."""
        dc_data = {}
        
        # Title
        title_elem = root.find('.//{http://purl.org/dc/elements/1.1/}title')
        if title_elem is not None:
            alt_elem = title_elem.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Alt')
            if alt_elem is not None:
                li = alt_elem.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li')
                if li is not None and li.text:
                    dc_data['title'] = li.text
        
        # Description/Caption
        desc_elem = root.find('.//{http://purl.org/dc/elements/1.1/}description')
        if desc_elem is not None:
            alt_elem = desc_elem.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Alt')
            if alt_elem is not None:
                li = alt_elem.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li')
                if li is not None and li.text:
                    dc_data['caption'] = li.text
        
        # Creator
        creator_elem = root.find('.//{http://purl.org/dc/elements/1.1/}creator')
        if creator_elem is not None:
            seq_elem = creator_elem.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Seq')
            if seq_elem is not None:
                li = seq_elem.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li')
                if li is not None and li.text:
                    dc_data['creator'] = li.text
        
        return dc_data
    
    def _extract_rights(self, root: ET.Element) -> Dict[str, Any]:
        """Extract rights metadata from XMP."""
        rights_data = {}
        
        description = root.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description')
        if description is None:
            return rights_data
        
        # Copyright
        copyright_elem = description.find('{http://ns.adobe.com/xap/1.0/rights/}Copyright')
        if copyright_elem is not None and copyright_elem.text:
            rights_data['copyright_notice'] = copyright_elem.text
        
        # Usage terms
        usage_elem = description.find('{http://ns.adobe.com/xap/1.0/rights/}UsageTerms')
        if usage_elem is not None and usage_elem.text:
            rights_data['rights_usage_terms'] = usage_elem.text
        
        return rights_data
    
    def _write_keywords(self, description: ET.Element, keywords: List[str]):
        """Write keywords to XMP."""
        # Remove existing subject element
        for elem in description.findall('{http://purl.org/dc/elements/1.1/}subject'):
            description.remove(elem)
        
        if not keywords:
            return
        
        # Create new subject element
        subject = ET.SubElement(description, '{http://purl.org/dc/elements/1.1/}subject')
        bag = ET.SubElement(subject, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Bag')
        
        for keyword in keywords:
            li = ET.SubElement(bag, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li')
            li.text = keyword
    
    def _write_iptc(self, description: ET.Element, iptc_data: Dict[str, Any]):
        """Write IPTC metadata to XMP."""
        # Map of IPTC fields to XMP elements
        field_map = {
            'sublocation': '{http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/}Location',
            'city': '{http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/}City',
            'region': '{http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/}State',
            'country': '{http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/}Country',
            'country_code': '{http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/}CountryCode',
            'headline': '{http://ns.adobe.com/photoshop/1.0/}Headline',
            'instructions': '{http://ns.adobe.com/photoshop/1.0/}Instructions',
            'credit': '{http://ns.adobe.com/photoshop/1.0/}Credit',
            'source': '{http://ns.adobe.com/photoshop/1.0/}Source',
        }
        
        for iptc_field, xmp_tag in field_map.items():
            if iptc_field in iptc_data:
                # Remove existing element
                for elem in description.findall(xmp_tag):
                    description.remove(elem)
                
                # Add new element
                elem = ET.SubElement(description, xmp_tag)
                elem.text = str(iptc_data[iptc_field])
    
    def _write_dublin_core(self, description: ET.Element, dc_data: Dict[str, Any]):
        """Write Dublin Core metadata to XMP."""
        # Title
        if 'title' in dc_data:
            for elem in description.findall('{http://purl.org/dc/elements/1.1/}title'):
                description.remove(elem)
            
            title = ET.SubElement(description, '{http://purl.org/dc/elements/1.1/}title')
            alt = ET.SubElement(title, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Alt')
            li = ET.SubElement(alt, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li')
            li.set('{http://www.w3.org/XML/1998/namespace}lang', 'x-default')
            li.text = dc_data['title']
        
        # Description/Caption
        if 'caption' in dc_data:
            for elem in description.findall('{http://purl.org/dc/elements/1.1/}description'):
                description.remove(elem)
            
            desc = ET.SubElement(description, '{http://purl.org/dc/elements/1.1/}description')
            alt = ET.SubElement(desc, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Alt')
            li = ET.SubElement(alt, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li')
            li.set('{http://www.w3.org/XML/1998/namespace}lang', 'x-default')
            li.text = dc_data['caption']
        
        # Creator
        if 'creator' in dc_data:
            for elem in description.findall('{http://purl.org/dc/elements/1.1/}creator'):
                description.remove(elem)
            
            creator = ET.SubElement(description, '{http://purl.org/dc/elements/1.1/}creator')
            seq = ET.SubElement(creator, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Seq')
            li = ET.SubElement(seq, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li')
            li.text = dc_data['creator']
    
    def _write_rights(self, description: ET.Element, rights_data: Dict[str, Any]):
        """Write rights metadata to XMP."""
        # Copyright
        if 'copyright_notice' in rights_data:
            for elem in description.findall('{http://ns.adobe.com/xap/1.0/rights/}Copyright'):
                description.remove(elem)
            
            copyright_elem = ET.SubElement(description, '{http://ns.adobe.com/xap/1.0/rights/}Copyright')
            copyright_elem.text = rights_data['copyright_notice']
        
        # Usage terms
        if 'rights_usage_terms' in rights_data:
            for elem in description.findall('{http://ns.adobe.com/xap/1.0/rights/}UsageTerms'):
                description.remove(elem)
            
            usage_elem = ET.SubElement(description, '{http://ns.adobe.com/xap/1.0/rights/}UsageTerms')
            usage_elem.text = rights_data['rights_usage_terms']
    
    def _prettify_xml(self, elem: ET.Element) -> str:
        """Return a pretty-printed XML string."""
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        
        # Add XMP packet wrapper
        xmp_header = '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        xmp_footer = '\n<?xpacket end="w"?>'
        
        pretty_xml = reparsed.toprettyxml(indent='  ', encoding=None)
        # Remove extra blank lines
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        
        return xmp_header + '\n'.join(lines) + xmp_footer


def sync_xmp_with_database(photo_path: str, db_metadata: Dict[str, Any]) -> bool:
    """Sync XMP sidecar file with database metadata."""
    try:
        xmp = XMPSidecar(photo_path)
        
        # Convert database metadata to XMP format
        xmp_metadata = {
            'keywords': [kw['keyword'] for kw in db_metadata.get('keywords', [])],
            'iptc': db_metadata.get('iptc', {}),
            'dc': {
                'title': db_metadata.get('iptc', {}).get('title'),
                'caption': db_metadata.get('iptc', {}).get('caption'),
                'creator': db_metadata.get('iptc', {}).get('creator'),
            },
            'rights': {
                'copyright_notice': db_metadata.get('iptc', {}).get('copyright_notice'),
                'rights_usage_terms': db_metadata.get('iptc', {}).get('rights_usage_terms'),
            }
        }
        
        # Remove None values
        xmp_metadata['dc'] = {k: v for k, v in xmp_metadata['dc'].items() if v is not None}
        xmp_metadata['rights'] = {k: v for k, v in xmp_metadata['rights'].items() if v is not None}
        
        return xmp.write(xmp_metadata)
        
    except Exception as e:
        logger.error(f"Failed to sync XMP for {photo_path}: {e}")
        return False


def read_xmp_metadata(photo_path: str) -> Dict[str, Any]:
    """Read metadata from XMP sidecar file."""
    try:
        xmp = XMPSidecar(photo_path)
        return xmp.read()
    except Exception as e:
        logger.error(f"Failed to read XMP for {photo_path}: {e}")
        return {}