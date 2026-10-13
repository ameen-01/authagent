"""Sentence banks and clinical vocabulary for the synthetic world.

Used by both the chart generator and the verifier training-pair generator.
Everything here is synthetic and simplified; it is not clinical guidance.

Each "core" sentence is tagged with the verifier label it carries for its criterion:
  SUPPORTS        -> evidence that the criterion is met
  REFUTES         -> evidence that the criterion is not met
  NOT_ENOUGH_INFO -> related-looking text that does not settle the criterion (near misses)
"""

from __future__ import annotations

GENERIC_DISTRACTORS = [
    "Blood pressure 128/82 mmHg, heart rate 76.",
    "No fever, chills, or recent infections reported.",
    "Social history: works in an office, does not smoke.",
    "Lives with family and is independent in daily activities.",
    "Influenza vaccine was administered this season.",
    "Medication list and allergies were reviewed.",
    "Sleep has been adequate and appetite is stable.",
    "Return to clinic in three months or sooner if needed.",
    "Discussed diet and regular physical activity.",
    "No known drug allergies.",
    "Weight is stable compared with the last visit.",
    "The patient agrees with the plan of care.",
    "Routine labs were reviewed with the patient.",
    "Denies chest pain or shortness of breath.",
]

PREFIXES = ["", "Per chart review, ", "Of note, ", "On review today, ", "Clinician documents: ", "History: "]
SUFFIXES = ["", " Will continue to monitor.", " Discussed with the patient.", " See prior notes for details."]

DOMAIN: dict[str, dict] = {
    "POL-RA": {
        "requested_drug": "adalimumab 40 mg every other week",
        "specialty": "Rheumatology",
        "dx_structured": [("M06.9", "Rheumatoid arthritis, unspecified"),
                          ("M05.79", "Rheumatoid arthritis with rheumatoid factor of multiple sites")],
        "dx_other": [("M17.11", "Unilateral primary osteoarthritis, right knee"),
                     ("M25.50", "Pain in unspecified joint")],
        "med": {"name": "methotrexate", "dose": "15 mg weekly"},
        "med_nearmiss": {"name": "hydroxychloroquine", "dose": "200 mg twice daily"},
        "lab": {"name": "QuantiFERON-TB Gold", "type": "categorical", "good": "negative", "bad": "positive",
                "recent_days": (10, 300), "stale_days": (400, 700)},
        "domain_distractors": [
            "Hand X-rays were ordered for baseline assessment.",
            "Folic acid 1 mg daily is continued.",
            "Counseled on infection risk with immunosuppressive therapy.",
            "Hepatitis B serologies are pending.",
            "Discussed joint protection and occupational therapy.",
        ],
        "cores": {
            "C1": {
                "SUPPORTS": [
                    "Assessment: seropositive rheumatoid arthritis with active synovitis of both hands.",
                    "Impression: established rheumatoid arthritis, polyarticular.",
                    "Known history of rheumatoid arthritis, anti-CCP positive.",
                    "Rheumatoid arthritis was confirmed by rheumatology based on serology and imaging.",
                    "Diagnosis: rheumatoid arthritis involving the wrists and MCP joints.",
                    "Long-standing rheumatoid arthritis followed in rheumatology clinic.",
                ],
                "REFUTES": [
                    "Rheumatoid arthritis has been ruled out; joint pain is attributed to osteoarthritis.",
                    "Serologies are negative and imaging does not support rheumatoid arthritis; findings are consistent with osteoarthritis.",
                    "Workup was negative for inflammatory arthritis and the diagnosis was revised to osteoarthritis.",
                    "Rheumatology consult concluded that the patient does not have rheumatoid arthritis.",
                ],
                "NOT_ENOUGH_INFO": [
                    "Family history of rheumatoid arthritis in the mother.",
                    "Rheumatoid factor test ordered to evaluate joint pain.",
                    "Referred to rheumatology to evaluate possible inflammatory arthritis.",
                    "Joint pain in the knees after prolonged walking.",
                ],
            },
            "C4": {
                "SUPPORTS": [
                    "Despite methotrexate, disease activity remains high with persistent synovitis.",
                    "Inadequate response to methotrexate after an adequate trial; joint swelling persists.",
                    "Methotrexate was discontinued due to elevated liver enzymes.",
                    "The patient could not tolerate methotrexate because of persistent nausea and mouth ulcers.",
                    "Ongoing active disease despite maximally tolerated methotrexate.",
                    "Methotrexate failed to control symptoms; tender and swollen joint counts remain elevated.",
                ],
                "REFUTES": [
                    "Disease is well controlled on methotrexate with minimal symptoms.",
                    "Good response to methotrexate; joint swelling has resolved.",
                    "Methotrexate is well tolerated and the patient reports marked improvement.",
                    "Low disease activity has been achieved on methotrexate monotherapy.",
                ],
                "NOT_ENOUGH_INFO": [
                    "Methotrexate was started at 15 mg weekly.",
                    "Inadequate response to sulfasalazine was documented previously.",
                    "The patient asked about side effects of methotrexate.",
                    "Methotrexate dosing and monitoring labs were reviewed.",
                ],
            },
        },
    },
    "POL-GLP1": {
        "requested_drug": "semaglutide 0.5 mg weekly",
        "specialty": "Endocrinology",
        "dx_structured": [("E11.9", "Type 2 diabetes mellitus without complications"),
                          ("E11.65", "Type 2 diabetes mellitus with hyperglycemia")],
        "dx_other": [("E10.9", "Type 1 diabetes mellitus without complications"),
                     ("R73.03", "Prediabetes")],
        "med": {"name": "metformin", "dose": "1000 mg twice daily"},
        "med_nearmiss": {"name": "glipizide", "dose": "5 mg daily"},
        "lab": {"name": "HbA1c", "type": "numeric", "unit": "%", "threshold": 7.0, "step": 0.1,
                "good_range": (7.2, 10.5), "bad_range": (5.8, 6.8),
                "recent_days": (5, 80), "stale_days": (100, 300)},
        "domain_distractors": [
            "Foot exam: sensation intact bilaterally.",
            "Referred for annual diabetic eye exam.",
            "Checks fasting glucose at home most mornings.",
            "Lipid panel was ordered.",
            "Discussed hypoglycemia symptoms and management.",
        ],
        "cores": {
            "C1": {
                "SUPPORTS": [
                    "Assessment: type 2 diabetes mellitus, suboptimally controlled.",
                    "Long-standing type 2 diabetes managed in primary care.",
                    "Known history of type 2 diabetes mellitus.",
                    "Diagnosis: type 2 diabetes with hyperglycemia.",
                    "Type 2 diabetes was diagnosed several years ago.",
                ],
                "REFUTES": [
                    "Diagnosis revised to type 1 diabetes based on positive GAD antibodies and low C-peptide.",
                    "The patient has type 1 diabetes managed with an insulin pump; there is no diagnosis of type 2 diabetes.",
                    "Glucose intolerance noted, but criteria for type 2 diabetes are not met.",
                    "Prediabetes only; the patient does not meet criteria for diabetes.",
                ],
                "NOT_ENOUGH_INFO": [
                    "Strong family history of type 2 diabetes in both parents.",
                    "Screening HbA1c ordered to evaluate for diabetes.",
                    "Discussed risk factors for developing diabetes.",
                    "Referred to a dietitian for weight management.",
                ],
            },
            "C5": {
                "SUPPORTS": [
                    "Denies personal or family history of medullary thyroid carcinoma or MEN2.",
                    "No personal or family history of medullary thyroid cancer.",
                    "Screened for thyroid cancer history: no medullary thyroid carcinoma in the patient or family.",
                    "Family history is negative for thyroid cancer, including medullary thyroid carcinoma.",
                    "No history of MEN2 or medullary thyroid carcinoma in the patient or relatives.",
                ],
                "REFUTES": [
                    "Family history is notable for medullary thyroid carcinoma in a sibling.",
                    "The patient has a history of medullary thyroid carcinoma treated with thyroidectomy.",
                    "Mother was diagnosed with MEN2 and medullary thyroid cancer.",
                    "Personal history of MEN2 syndrome.",
                ],
                "NOT_ENOUGH_INFO": [
                    "Thyroid exam is normal without goiter or nodules.",
                    "TSH is within normal limits.",
                    "History of hypothyroidism on levothyroxine.",
                    "Thyroid ultrasound was ordered for a palpable nodule.",
                ],
            },
        },
    },
    "POL-PCSK9": {
        "requested_drug": "evolocumab 140 mg every two weeks",
        "specialty": "Cardiology",
        "dx_structured": [("I25.10", "Atherosclerotic heart disease of native coronary artery"),
                          ("E78.01", "Familial hypercholesterolemia")],
        "dx_other": [("E78.5", "Hyperlipidemia, unspecified"),
                     ("I10", "Essential hypertension")],
        "med": {"name": "atorvastatin", "dose": "80 mg daily"},
        "med_nearmiss": {"name": "pravastatin", "dose": "40 mg daily"},
        "lab": {"name": "LDL-C", "type": "numeric", "unit": "mg/dL", "threshold": 70, "step": 1,
                "good_range": (75, 190), "bad_range": (45, 65),
                "recent_days": (5, 110), "stale_days": (130, 400)},
        "domain_distractors": [
            "Walks thirty minutes most days.",
            "Echocardiogram last year showed preserved ejection fraction.",
            "Counseled on a Mediterranean-style diet.",
            "Blood pressure is controlled on the current regimen.",
            "Advised to limit saturated fat intake.",
        ],
        "cores": {
            "C1": {
                "SUPPORTS": [
                    "History of coronary artery disease with prior myocardial infarction.",
                    "Assessment: atherosclerotic cardiovascular disease, status post coronary stenting.",
                    "Heterozygous familial hypercholesterolemia was confirmed by genetic testing.",
                    "Established coronary artery disease with stable angina.",
                    "Prior ischemic stroke attributed to atherosclerotic disease.",
                ],
                "REFUTES": [
                    "No history of coronary, cerebrovascular, or peripheral arterial disease; genetic testing was negative for familial hypercholesterolemia.",
                    "Primary prevention patient without established cardiovascular disease or familial hypercholesterolemia.",
                    "Coronary angiography showed no significant disease, and familial hypercholesterolemia was excluded.",
                ],
                "NOT_ENOUGH_INFO": [
                    "Father had a myocardial infarction at age 58.",
                    "Coronary calcium score was ordered.",
                    "Discussed cardiovascular risk factors.",
                    "Referred to cardiology for risk assessment.",
                ],
            },
            "C5": {
                "SUPPORTS": [
                    "Reports taking atorvastatin daily without missed doses; pharmacy refills are on time.",
                    "Adherent to statin therapy per patient report and refill history.",
                    "Statin adherence confirmed with no missed doses in the past six months.",
                    "Takes the statin consistently as prescribed.",
                ],
                "REFUTES": [
                    "Admits to frequently skipping statin doses over the past several months.",
                    "Refill history shows long gaps; the patient has not been taking the statin regularly.",
                    "The patient stopped the statin without medical advice several months ago and has not resumed.",
                ],
                "NOT_ENOUGH_INFO": [
                    "Statin was prescribed at the last visit.",
                    "Discussed the importance of medication adherence in general.",
                    "Reports good adherence to blood pressure medication.",
                    "Statin side effects were reviewed.",
                ],
            },
        },
    },
}

# Criterion phrasings used to vary the verifier's input during training.
CRITERION_PARAPHRASES: dict[tuple[str, str], list[str]] = {
    ("POL-RA", "C1"): ["Patient has a diagnosis of rheumatoid arthritis.",
                       "The patient is diagnosed with rheumatoid arthritis.",
                       "Rheumatoid arthritis diagnosis is established."],
    ("POL-RA", "C4"): ["Inadequate response to, or intolerance of, methotrexate is documented.",
                       "Methotrexate failed or was not tolerated.",
                       "Documentation shows methotrexate did not work or caused intolerable side effects."],
    ("POL-GLP1", "C1"): ["Patient has a diagnosis of type 2 diabetes mellitus.",
                         "The patient is diagnosed with type 2 diabetes.",
                         "Type 2 diabetes mellitus diagnosis is established."],
    ("POL-GLP1", "C5"): ["Absence of personal or family history of medullary thyroid carcinoma is documented.",
                         "The patient and family have no history of medullary thyroid carcinoma.",
                         "No medullary thyroid cancer or MEN2 history is documented."],
    ("POL-PCSK9", "C1"): ["Patient has a diagnosis of atherosclerotic cardiovascular disease or heterozygous familial hypercholesterolemia.",
                          "The patient has established ASCVD or heterozygous familial hypercholesterolemia.",
                          "Clinical ASCVD or HeFH is diagnosed."],
    ("POL-PCSK9", "C5"): ["Adherence to statin therapy is documented.",
                          "The patient takes the statin as prescribed.",
                          "Documentation shows the patient is adherent to statin therapy."],
}

FIRST_NAMES = ["Kavya", "Arjun", "Maria", "James", "Priya", "Daniel", "Aisha", "Rohan", "Elena", "Samuel",
               "Meera", "Thomas", "Fatima", "Vikram", "Grace", "Ibrahim", "Ananya", "Lucas", "Nisha", "Robert"]
LAST_NAMES = ["Iyer", "Sharma", "Garcia", "Walker", "Nair", "Chen", "Khan", "Mehta", "Rossi", "Okafor",
              "Reddy", "Miller", "Haddad", "Menon", "Lopez", "Pillai", "Kim", "Fernandes", "Das", "Wright"]
STREETS = ["Lakeview Rd", "Maple Ave", "Station St", "Hillcrest Dr", "Park Lane", "River Rd"]
CITIES = ["Springfield", "Riverton", "Fairview", "Lakeside", "Brookfield"]
