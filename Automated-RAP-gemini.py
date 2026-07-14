import os
import json
import time
from datetime import datetime
from google import genai
import google.auth
from google.genai import types
from google.cloud import storage

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "gemini-2.5-pro"
BUCKET_NAME = "knowledge_extraction"
LOCATION = "us-central1"

# Automatically create local folders for outputs
os.makedirs("agent1_pared_outputs", exist_ok=True)
os.makedirs("agent2_final_outputs", exist_ok=True)

# GCS Prefixes (using timestamps to avoid overwriting previous runs)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
GCS_INPUT_DIR = f"gs://{BUCKET_NAME}/inputs_{run_id}/"
GCS_OUTPUT_DIR_AGENT1 = f"gs://{BUCKET_NAME}/batch_results_agent1_{run_id}/"
GCS_OUTPUT_DIR_AGENT2 = f"gs://{BUCKET_NAME}/batch_results_agent2_{run_id}/"

# ==========================================
# AUTHENTICATION
# ==========================================
credentials, project_id = google.auth.default()
client = genai.Client(
    vertexai=True,
    project=project_id,
    location=LOCATION,
    credentials=credentials
)
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_file_to_string(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def fill_prompt_template(template_text, values_dict):
    for key, value in values_dict.items():
        template_text = template_text.replace(f"{{{key}}}", value)
    return template_text

def wait_for_job(job_name, client):
    """Polls the Vertex AI job until it succeeds or fails."""
    print(f"Polling job {job_name}...")
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name
        if state == 'JOB_STATE_SUCCEEDED':
            print(f"\nJob {job_name} SUCCEEDED!")
            return True
        elif state in ['JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_PARTIALLY_SUCCEEDED']:
            print(f"\nJob {job_name} stopped with state: {state}")
            if job.error:
                 print(f"Error: {job.error}")
            return False
        print(".", end="", flush=True)
        time.sleep(30) # Wait 30 seconds between checks

# ==========================================
# PROMPTS
# ==========================================
agent1_sys_schema = """
You are an expert knowledge extraction analyst.

You are given:
 - A set of schema triples in the format (subject, predicate, object), which define a knowledge schema
 - A scientific publication or report in markdown format
 
Your task is to review the schema and determine which triples are realistically populatable based on the kind of information found in the publication — not to extract the actual data values.

Only include triples for which:
 - The publication clearly discusses or covers the subject matter described in the triples
 - There is a strong indication that data could be extracted to populate the triples, even if not extracted now

Do not include:
 - Triples that refer to topics or entities not mentioned at all in the publication
 - Triples for which the necessary information is missing or unlikely to be derivable from the content

Output only the subset of relevant schema triples — the ones that appear potentially populatable — in CSV format.
"""

agent1_usr_schema = """
Task: Read the publication and filter the schema triples. Output only those triples for which there is evidence that the information is likely present or derivable from the publication.

Do not extract or fill in the data. Only identify which schema triples are realistically populatable based on content coverage.

Output format:
A CSV list of supported triples in this format:
"subject,predicate,object"

Schema: \"\"\"
{schema}
\"\"\"

Publication: \"\"\"
{publication}
\"\"\"
"""

agent1_sys_lit = """
You are an expert knowledge extraction analyst.

You are given:
 - A set of schema triples in the format (subject, predicate, object), which define a knowledge schema
 - A scientific publication or report in markdown format
 
Your task is to review the schema and extract pieces of the publication that can realistically populate the given schema triples.

Only include the publication data extract for which:
 - The publication clearly discusses or covers the subject matter described in the triples
 - There is a strong indication that data could be extracted to populate the triples, even if not extracted now

Do not:
 - Change or rephrase anything in the publication but keep the wording as text as it is.
 - include data extracts for which the necessary information is missing or unlikely to be derivable from the content

Output only the subset publication extract — in markdown format.
"""

agent1_usr_lit = """
Task: Read the publication and the schema triples. Output only those publication extracts for which there is evidence that the information is likely present or derivable and can populate the triples.

Do not change or rephrase the publication wording but give exact same wording. Only identify the sub-extracts of the publication that can realistically populate triples based on content coverage.

Output format:
A markdown of the exact extracts of the publication:

Schema: \"\"\"
{schema}
\"\"\"

Publication: \"\"\"
{publication}
\"\"\"
"""

agent2_sys_no_ex = """
You are an expert in precise information extraction.

You are given:
 - A list of schema triples in the form: (subject, predicate, object) — each defines a relation pattern that is expected to be supported by the publication
 - A publication in markdown format

Your task is to:
 - Carefully read the publication
 - Extract all factual data triples from the publication that match the patterns defined in the schema
 - Only extract triples that are explicitly stated or clearly inferable from the publication
 - Return only the final data triples extracted

Guidelines:
 - Only include triples that adhere to the schema exactly
 - Do not include unsupported, hypothetical, or guessed information
 - If multiple valid triples fit the same schema (e.g., multiple values), include them all
 - Do not include input schema triples but only include the extracted data triples
 - Output must be a CSV-style list of complete data triples: "subject,predicate,object"
"""

agent2_sys_ex = """
You are an expert in precise information extraction.

You are given:
 - A list of schema triples in the form: (subject, predicate, object) — each defines a relation pattern that is expected to be supported by the publication
 - A publication in markdown format

Your task is to:
 - Carefully read the publication
 - Analyze the provided examples for populating triples with data
 - Extract all factual data triples from the publication that match the patterns defined in the schema using the examples as a guide
 - Only extract triples that are explicitly stated or clearly inferable from the publication
 - Return only the final data triples extracted

Guidelines:
 - Only include triples that adhere to the schema exactly
 - Do not include unsupported, hypothetical, or guessed information
 - If multiple valid triples fit the same schema (e.g., multiple values), include them all
 - Do not include input schema triples but only include the extracted data triples
 - Output must be a CSV-style list of complete data triples: "subject,predicate,object"
"""

agent2_usr_base = """
Task: From the publication, extract all factual data triples that match the patterns defined in the schema.

Only extract triples if they are explicitly stated or clearly inferable. Return one CSV-formatted triple per line.
{examples_block}
Format:
"subject,predicate,object"

Schema: \"\"\"
{schema}
\"\"\"

Publication: \"\"\"
{publication}
\"\"\"
"""

EXAMPLES_TEXT = """
Below are examples of what a triple in the schema could become when populated with data.

Examples:
1. Material,hasProperty,Property -> Polyethylene,hasProperty,Density: 0.95 g/cm³
2. Experiment,hasLocation,Location -> Experiment001,hasLocation,MIT Laboratory 3B
3. DifferentialScanningCalorimetry,hasCalorimetryCharacteristic,CoolingRate -> DSC_Exp12,hasCalorimetryCharacteristic,CoolingRate: 10 °C/min
4. Rheometry,hasRheometryCharacteristic,CapillarySize -> RheoTest22,hasRheometryCharacteristic,CapillarySize: 0.5 mm
5. TransmissionElectronMicroscopy,hasMicrscopyCharacteristic,Magnification -> TEM_SampleA,hasMicrscopyCharacteristic,Magnification: 50000x
6. Collection,hasDOI,xsd:anyURI -> NanocompositeStudy2023,hasDOI,https://doi.org/10.1016/j.polymer.2023.125
7. Reference,hasAuthor,Author -> Ref1234,hasAuthor,Jane Smith
8. Material,hasComponent,Material -> EpoxyResinBlend,hasComponent,CarbonNanotubes
9. Computation,hasSoftwareConfiguration,SoftwareConfiguration -> MDRun45,hasSoftwareConfiguration,LAMMPS v3.2
10. Data,hasSamplePreparation,PhysicalProcess -> DataSet_AlFoam,hasSamplePreparation,HeatTreatment950C
11. CharacterizationMethod,hasResultData,ResultData -> SEM_Char2022,hasResultData,SEM_Results_Char22.csv
12. FiberTensileStrength,isMeasuredUnderCondition,Conditions -> FiberX_2021,isMeasuredUnderCondition,Temperature: 25 °C
13. TensileModulus,isMeasuredUnderCondition,Conditions -> PLA_Sample7,isMeasuredUnderCondition,StrainRate: 0.01 /s
14. Material,hasProperty,SpecificSurfaceArea -> SiO2_Powder_A,hasProperty,SSA_SiO2_A
15. SpecificSurfaceArea,hasQuantity,Value -> SSA_SiO2_A,hasQuantity,200 m²/g
16. Additive,hasQuantity,Quantity -> TiO2_Nanoparticles,hasQuantity,3 wt%
"""

# ==========================================
# MAIN PIPELINE EXECUTION
# ==========================================
def run_pipeline():
    # 1. Load static files
    pub_string = load_file_to_string("publication.md")
    
    # Define combinations (Matches your 8 experiment conditions)
    experiments = [
        {"schema_complete": True,  "s_par": True,  "l_par": False, "e_prov": False},
        {"schema_complete": True,  "s_par": False, "l_par": True,  "e_prov": False},
        {"schema_complete": True,  "s_par": True,  "l_par": False, "e_prov": True},
        {"schema_complete": True,  "s_par": False, "l_par": True,  "e_prov": True},
        {"schema_complete": False, "s_par": True,  "l_par": False, "e_prov": False},
        {"schema_complete": False, "s_par": False, "l_par": True,  "e_prov": False},
        {"schema_complete": False, "s_par": True,  "l_par": False, "e_prov": True},
        {"schema_complete": False, "s_par": False, "l_par": True,  "e_prov": True},
    ]

    # =======================================================
    # STAGE 1: AGENT 1 (SCHEMA / LIT PARING)
    # =======================================================
    print("--- GENERATING BATCH 1 (AGENT 1) ---")
    json_requests_agent1 = []

    for exp in experiments:
        s_comp, s_par, l_par, e_prov = exp["schema_complete"], exp["s_par"], exp["l_par"], exp["e_prov"]
        
        # Determine which schemas to iterate over
        if s_comp:
            schemas_to_process = {"triples": load_file_to_string("triples.csv")}
        else:
            schemas_to_process = {}
            for filename in sorted(os.listdir("csv")):
                if filename.endswith(".csv"):
                    base_name = os.path.splitext(filename)[0]
                    schemas_to_process[base_name] = load_file_to_string(os.path.join("csv", filename))
                    
        # Build requests
        for schema_name, schema_content in schemas_to_process.items():
            if s_par and not l_par:
                sys_msg, usr_msg = agent1_sys_schema, agent1_usr_schema
            elif not s_par and l_par:
                sys_msg, usr_msg = agent1_sys_lit, agent1_usr_lit
                
            prompt = fill_prompt_template(usr_msg, {"publication": pub_string, "schema": schema_content})
            key = f"agent1-{s_comp}-{s_par}-{l_par}-{e_prov}-{schema_name}"
            
            json_requests_agent1.append({
                "key": key,
                "request": {
                    "system_instruction": {"parts": [{"text": sys_msg}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}]
                }
            })

    # Write & Upload Agent 1 Batch
    agent1_file = "agent1_batch.jsonl"
    with open(agent1_file, 'w', encoding='utf-8') as f:
        for req in json_requests_agent1:
            f.write(json.dumps(req) + '\n')

    print(f"Uploading {agent1_file} to GCS...")
    bucket.blob(f"inputs_{run_id}/{agent1_file}").upload_from_filename(agent1_file)

    print("Triggering Vertex AI Batch Job 1...")
    job1 = client.batches.create(
        model=MODEL_NAME,
        src=f"{GCS_INPUT_DIR}{agent1_file}",
        config=types.CreateBatchJobConfig(
            dest=GCS_OUTPUT_DIR_AGENT1,
            display_name="auto-rap-agent1"
        )
    )
    
    # Wait for completion
    if not wait_for_job(job1.name, client):
        print("Agent 1 Job Failed. Exiting.")
        return

    # =======================================================
    # STAGE 2: PARSE AGENT 1 & BUILD AGENT 2 BATCH
    # =======================================================
    print("\n--- GENERATING BATCH 2 (AGENT 2) ---")
    json_requests_agent2 = []
    
    # We need to map the original schemas/publications to build Agent 2's prompt properly
    original_schemas = {"triples": load_file_to_string("triples.csv")}
    for filename in os.listdir("csv"):
        if filename.endswith(".csv"):
            original_schemas[os.path.splitext(filename)[0]] = load_file_to_string(os.path.join("csv", filename))

    # Fetch results from GCS
    blobs = bucket.list_blobs(prefix=f"batch_results_agent1_{run_id}/")
    for blob in blobs:
        if blob.name.endswith('.jsonl'):
            content = blob.download_as_text()
            for line in content.strip().split('\n'):
                if not line: continue
                data = json.loads(line)
                
                if "candidates" in data.get("response", {}):
                    agent1_output = data["response"]["candidates"][0]["content"]["parts"][0]["text"].strip()
                    req_key = data.get("key", "")
                    
                    # Save Agent 1 output locally for auditing
                    with open(os.path.join("agent1_pared_outputs", f"{req_key}.txt"), 'w') as f:
                        f.write(agent1_output)
                        
                    # Parse the key to know the configuration
                    # Format: agent1-{s_comp}-{s_par}-{l_par}-{e_prov}-{schema_name}
                    parts = req_key.split('-')
                    s_comp, s_par, l_par, e_prov, schema_name = parts[1], parts[2], parts[3], parts[4], parts[5]
                    
                    # Determine Agent 2 inputs based on Agent 1 logic
                    if s_par == 'True' and l_par == 'False':
                        # Schema was pared, lit stays the same
                        current_schema = agent1_output
                        current_pub = pub_string
                    else:
                        # Lit was pared, schema stays the same
                        current_schema = original_schemas[schema_name]
                        current_pub = agent1_output
                        
                    # Prepare Agent 2 Prompt
                    sys_msg = agent2_sys_ex if e_prov == 'True' else agent2_sys_no_ex
                    ex_block = f"\n{EXAMPLES_TEXT}\n" if e_prov == 'True' else ""
                    
                    prompt = fill_prompt_template(agent2_usr_base, {
                        "examples_block": ex_block,
                        "schema": current_schema,
                        "publication": current_pub
                    })
                    
                    agent2_key = f"agent2-{s_comp}-{s_par}-{l_par}-{e_prov}-{schema_name}"
                    json_requests_agent2.append({
                        "key": agent2_key,
                        "request": {
                            "system_instruction": {"parts": [{"text": sys_msg}]},
                            "contents": [{"role": "user", "parts": [{"text": prompt}]}]
                        }
                    })

    # Write & Upload Agent 2 Batch
    agent2_file = "agent2_batch.jsonl"
    with open(agent2_file, 'w', encoding='utf-8') as f:
        for req in json_requests_agent2:
            f.write(json.dumps(req) + '\n')

    print(f"Uploading {agent2_file} to GCS...")
    bucket.blob(f"inputs_{run_id}/{agent2_file}").upload_from_filename(agent2_file)

    print("Triggering Vertex AI Batch Job 2...")
    job2 = client.batches.create(
        model=MODEL_NAME,
        src=f"{GCS_INPUT_DIR}{agent2_file}",
        config=types.CreateBatchJobConfig(
            dest=GCS_OUTPUT_DIR_AGENT2,
            display_name="auto-rap-agent2"
        )
    )
    
    # Wait for completion
    if not wait_for_job(job2.name, client):
        print("Agent 2 Job Failed. Exiting.")
        return

    # =======================================================
    # STAGE 3: PARSE FINAL RESULTS
    # =======================================================
    print("\n--- PROCESSING FINAL EXTRACTS ---")
    blobs = bucket.list_blobs(prefix=f"batch_results_agent2_{run_id}/")
    for blob in blobs:
        if blob.name.endswith('.jsonl'):
            content = blob.download_as_text()
            for line in content.strip().split('\n'):
                if not line: continue
                data = json.loads(line)
                
                if "candidates" in data.get("response", {}):
                    final_extract = data["response"]["candidates"][0]["content"]["parts"][0]["text"].strip()
                    req_key = data.get("key", "")
                    
                    # Output final CSVs matching original format
                    # e.g., gemini-2.5-pro-True-True-False-False-knowledge-extraction.csv
                    # req_key is 'agent2-True-True-False-False-triples'
                    parts = req_key.split('-')
                    filename = f"{MODEL_NAME}-{parts[1]}-{parts[2]}-{parts[3]}-{parts[4]}-{parts[5]}-knowledge-extraction.csv"
                    
                    with open(os.path.join("agent2_final_outputs", filename), 'w') as f:
                        f.write(final_extract)
                        
    print(f"\nPipeline complete! All results saved in 'agent2_final_outputs' directory.")

if __name__ == "__main__":
    run_pipeline()