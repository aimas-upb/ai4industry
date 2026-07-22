import httpx
import rdflib
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS
from typing import Optional

from src.config import EXAMPLES_DIR

# Define namespaces
TD = Namespace("https://www.w3.org/2019/wot/td#")
HCTL = Namespace("https://www.w3.org/2019/wot/hypermedia#")
ONTO = Namespace("https://ci.mines-stetienne.fr/kg/ontology#")
JSONSCHEMA = Namespace("https://www.w3.org/2019/wot/json-schema#")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
SSN = Namespace("http://www.w3.org/ns/ssn/")
S4SYST = Namespace("https://saref.etsi.org/saref4syst/")
CDT = Namespace("https://w3id.org/cdt/")
DCT = Namespace("http://purl.org/dc/terms/")


def fetch_graph(artifact_name: str) -> Optional[Graph]:
    """
    Fetch RDF graph for an artifact by name.
    Tries to fetch from remote KG using standard URI pattern, falls back to local examples/.ttl file.
    """
    graph = Graph()

    # Bind namespaces
    graph.bind("td", TD)
    graph.bind("hctl", HCTL)
    graph.bind("onto", ONTO)
    graph.bind("jsonschema", JSONSCHEMA)
    graph.bind("sosa", SOSA)
    graph.bind("ssn", SSN)
    graph.bind("s4syst", S4SYST)
    graph.bind("cdt", CDT)
    graph.bind("dct", DCT)

    # Try to fetch from remote using standard artifact URI pattern
    try:
        uri = f"https://ci.mines-stetienne.fr/kg/itmfactory/{artifact_name.lower()}#this"
        response = httpx.get(uri, timeout=5)
        response.raise_for_status()
        graph.parse(data=response.text, format="turtle")
        return graph
    except Exception:
        pass

    # Fallback: try local file
    local_file = EXAMPLES_DIR / f"{artifact_name.lower()}.ttl"
    if local_file.exists():
        try:
            graph.parse(local_file, format="turtle")
            return graph
        except Exception:
            pass

    return None


def get_thing_description(graph: Graph, thing_uri: str) -> dict:
    """Return basic Thing metadata: title, description, semantic types."""
    thing = rdflib.URIRef(thing_uri)
    result = {"uri": thing_uri}

    # Get title
    for obj in graph.objects(thing, RDFS.label):
        result["title"] = str(obj)
        break

    # Get comment
    for obj in graph.objects(thing, RDFS.comment):
        result["description"] = str(obj)
        break

    # Get rdf:type (semantic types)
    types = []
    for obj in graph.objects(thing, RDF.type):
        types.append(str(obj))
    result["types"] = types

    return result


def list_action_affordances(graph: Graph, thing_uri: str) -> list[dict]:
    """
    Extract all td:hasActionAffordance entries.
    Returns list of {name, description, semantic_type, endpoint_url, input_schema}.
    """
    thing = rdflib.URIRef(thing_uri)
    actions = []

    for aff in graph.objects(thing, TD.hasActionAffordance):
        action = {}

        # Get name
        for name_obj in graph.objects(aff, TD.name):
            action["name"] = str(name_obj)
            break

        # Get natural-language description (dct:description on the affordance node)
        action["description"] = ""
        for desc_obj in graph.objects(aff, DCT.description):
            action["description"] = str(desc_obj)
            break

        # Get semantic type (rdf:type of affordance)
        types = []
        for type_obj in graph.objects(aff, RDF.type):
            type_str = str(type_obj)
            if "onto:" in type_str or "#" in type_str:
                types.append(type_str.split("#")[-1] if "#" in type_str else type_str.split("/")[-1])
        action["semantic_type"] = types[0] if types else "Action"

        # Get endpoint URL (hctl:hasTarget from form)
        for form in graph.objects(aff, TD.hasForm):
            for target in graph.objects(form, HCTL.hasTarget):
                action["endpoint_url"] = str(target)
                break

        # Get input schema (jsonschema properties)
        schema = {}
        for form in graph.objects(aff, TD.hasForm):
            # Look for jsonschema:properties
            for prop_obj in graph.objects(form, JSONSCHEMA.properties):
                # This is typically a blank node with properties
                schema = _extract_schema(graph, prop_obj)
                break
        action["input_schema"] = schema
        action["op_type"] = "invokeaction"

        actions.append(action)

    return actions


def list_property_affordances(graph: Graph, thing_uri: str) -> list[dict]:
    """
    Extract all td:hasPropertyAffordance entries.
    Returns list of {name, description, semantic_type, endpoint_url, op_type, schema}.
    """
    thing = rdflib.URIRef(thing_uri)
    properties = []

    for aff in graph.objects(thing, TD.hasPropertyAffordance):
        prop = {}

        # Get name
        for name_obj in graph.objects(aff, TD.name):
            prop["name"] = str(name_obj)
            break

        # Get natural-language description (dct:description on the affordance node)
        prop["description"] = ""
        for desc_obj in graph.objects(aff, DCT.description):
            prop["description"] = str(desc_obj)
            break

        # Get semantic type
        types = []
        for type_obj in graph.objects(aff, RDF.type):
            type_str = str(type_obj)
            if "onto:" in type_str or "#" in type_str:
                types.append(type_str.split("#")[-1] if "#" in type_str else type_str.split("/")[-1])
        prop["semantic_type"] = types[0] if types else "Property"

        # Get endpoint URL
        for form in graph.objects(aff, TD.hasForm):
            for target in graph.objects(form, HCTL.hasTarget):
                prop["endpoint_url"] = str(target)
                break

            # Get operation type (readproperty / writeproperty)
            op_type = "readproperty"
            for op in graph.objects(form, HCTL.hasOperationType):
                op_type = str(op).split("#")[-1] if "#" in str(op) else str(op)
            prop["op_type"] = op_type

        # Get output schema type
        schema = {}
        for form in graph.objects(aff, TD.hasForm):
            # Look for schema references
            for schema_ref in graph.objects(form, JSONSCHEMA.type):
                schema["type"] = str(schema_ref)
            for enum_vals in graph.objects(form, JSONSCHEMA.enum):
                # enum might be a list
                schema["enum"] = str(enum_vals)
        prop["schema"] = schema

        properties.append(prop)

    return properties


def get_location_info(graph: Graph, thing_uri: str) -> dict:
    """
    Extract spatial info: artifact origin, relative areas, and absolute product locations.

    Returns:
        dict with keys:
        - origin: artifact base coordinates (onto:hasOriginCoordinates)
        - relative_input_area: relative coordinates of input area (onto:inputArea -> onto:relativeCoordinates)
        - relative_output_area: relative coordinates of output area (onto:outputArea -> onto:relativeCoordinates)
        - product_input_location: absolute coordinates where input material arrives (onto:locationOfInputMaterial)
        - product_output_location: absolute coordinates where output product departs (onto:locationOfOutputProduct)
    """
    thing = rdflib.URIRef(thing_uri)
    location = {}

    def extract_coords(node) -> dict:
        """Extract x, y, z coordinates from a location node."""
        coords = {}
        for x in graph.objects(node, ONTO.coordX):
            coords["x"] = float(x) if _is_number(str(x)) else str(x)
        for y in graph.objects(node, ONTO.coordY):
            coords["y"] = float(y) if _is_number(str(y)) else str(y)
        for z in graph.objects(node, ONTO.coordZ):
            coords["z"] = float(z) if _is_number(str(z)) else str(z)
        return coords

    # Get artifact origin coordinates
    for origin in graph.objects(thing, ONTO.hasOriginCoordinates):
        coords = extract_coords(origin)
        if coords:
            location["origin"] = coords

    # Get relative input area coordinates
    for input_area in graph.objects(thing, ONTO.inputArea):
        for rel_coords in graph.objects(input_area, ONTO.relativeCoordinates):
            coords = extract_coords(rel_coords)
            if coords:
                location["relative_input_area"] = coords
                break

    # Get relative output area coordinates
    for output_area in graph.objects(thing, ONTO.outputArea):
        for rel_coords in graph.objects(output_area, ONTO.relativeCoordinates):
            coords = extract_coords(rel_coords)
            if coords:
                location["relative_output_area"] = coords
                break

    # Get absolute product input location
    for input_loc in graph.objects(thing, ONTO.locationOfInputMaterial):
        coords = extract_coords(input_loc)
        if coords:
            location["product_input_location"] = coords

    # Get absolute product output location
    for output_loc in graph.objects(thing, ONTO.locationOfOutputProduct):
        coords = extract_coords(output_loc)
        if coords:
            location["product_output_location"] = coords

    return location


def _extract_schema(graph: Graph, schema_node) -> dict:
    """Extract JSON schema from a schema node."""
    schema = {}
    node = rdflib.URIRef(schema_node) if not isinstance(schema_node, rdflib.URIRef) else schema_node

    for type_obj in graph.objects(node, JSONSCHEMA.type):
        schema["type"] = str(type_obj)

    properties = {}
    for prop in graph.objects(node, JSONSCHEMA.properties):
        for prop_name in graph.objects(prop, RDFS.label):
            prop_type = None
            for t in graph.objects(prop, JSONSCHEMA.type):
                prop_type = str(t)
            if prop_type:
                properties[str(prop_name)] = {"type": prop_type}

    if properties:
        schema["properties"] = properties

    return schema


def _is_number(s: str) -> bool:
    """Check if string is a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False
