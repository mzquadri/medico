"""
Chest X-Ray Disease Detection - Disease Information Database

This module contains medical information about thoracic diseases
detected by the model. Information is intended for educational
purposes and clinical decision support.

Note: This information is not a substitute for professional
medical diagnosis. Always consult a qualified healthcare provider.
"""


DISEASE_INFO = {
    "Atelectasis": {
        "full_name": "Atelectasis (Lung Collapse)",
        "description": (
            "Atelectasis refers to the partial or complete collapse of lung tissue, "
            "resulting in reduced or absent gas exchange. It occurs when the alveoli "
            "(small air sacs) deflate or fill with fluid. This condition can affect "
            "part of a lobe, an entire lobe, or the whole lung."
        ),
        "common_locations": [
            "Lower lobes (most common)",
            "Middle lobe syndrome (right lung)",
            "Post-surgical regions",
            "Areas adjacent to pleural effusions"
        ],
        "radiographic_findings": [
            "Increased opacity in the affected region",
            "Volume loss with shift of mediastinum toward affected side",
            "Elevation of the hemidiaphragm",
            "Displacement of fissures",
            "Compensatory hyperinflation of adjacent lung"
        ],
        "causes": [
            "Airway obstruction (mucus plug, tumor, foreign body)",
            "External compression (pleural effusion, pneumothorax)",
            "Post-surgical (especially thoracic/abdominal surgery)",
            "Prolonged bed rest or shallow breathing",
            "Surfactant deficiency",
            "Chest trauma"
        ],
        "risk_factors": [
            "Recent surgery under general anesthesia",
            "Prolonged immobility",
            "Underlying lung disease",
            "Smoking history",
            "Obesity",
            "Neuromuscular disorders"
        ],
        "precautions": [
            "Early mobilization after surgery",
            "Incentive spirometry exercises",
            "Deep breathing and coughing exercises",
            "Adequate pain control to enable deep breathing",
            "Chest physiotherapy when indicated",
            "Bronchoscopy if obstruction suspected"
        ],
        "clinical_significance": "Moderate to High",
        "urgency": "Variable - depends on extent and underlying cause"
    },
    
    "Cardiomegaly": {
        "full_name": "Cardiomegaly (Enlarged Heart)",
        "description": (
            "Cardiomegaly is an enlargement of the heart visible on chest radiograph. "
            "It indicates the heart is larger than normal, typically defined as a "
            "cardiothoracic ratio greater than 0.5 on a PA chest X-ray. This finding "
            "often reflects underlying cardiac disease."
        ),
        "common_locations": [
            "Cardiac silhouette enlargement (global)",
            "Left ventricular enlargement (apex displaced laterally)",
            "Right ventricular enlargement (filling of retrosternal space)",
            "Left atrial enlargement (double density sign, splaying of carina)"
        ],
        "radiographic_findings": [
            "Cardiothoracic ratio greater than 50%",
            "Displacement of the cardiac apex",
            "Increased transverse cardiac diameter",
            "Straightening or convexity of left heart border",
            "Double density sign (left atrial enlargement)"
        ],
        "causes": [
            "Hypertension (most common)",
            "Coronary artery disease",
            "Valvular heart disease",
            "Cardiomyopathy (dilated, hypertrophic)",
            "Congenital heart disease",
            "Pericardial effusion (pseudo-cardiomegaly)",
            "Heart failure"
        ],
        "risk_factors": [
            "Uncontrolled hypertension",
            "History of myocardial infarction",
            "Diabetes mellitus",
            "Obesity",
            "Family history of heart disease",
            "Alcohol abuse",
            "Thyroid disorders"
        ],
        "precautions": [
            "Blood pressure control and monitoring",
            "Medication adherence (ACE inhibitors, beta-blockers, diuretics)",
            "Salt restriction in diet",
            "Regular cardiovascular follow-up",
            "Echocardiography for further evaluation",
            "Lifestyle modifications (exercise, weight management)",
            "Smoking cessation"
        ],
        "clinical_significance": "High",
        "urgency": "Requires cardiovascular evaluation"
    },
    
    "Consolidation": {
        "full_name": "Pulmonary Consolidation",
        "description": (
            "Consolidation refers to a region of lung tissue that has filled with "
            "liquid, pus, blood, cells, or other material instead of air. This results "
            "in increased density on chest X-ray. The most common cause is pneumonia, "
            "where the alveoli fill with inflammatory exudate."
        ),
        "common_locations": [
            "Lower lobes (aspiration, bacterial pneumonia)",
            "Upper lobes (tuberculosis, Klebsiella)",
            "Posterior segments (aspiration in supine patients)",
            "Lobar distribution (typical bacterial pneumonia)",
            "Patchy/multifocal (atypical pneumonia, viral)"
        ],
        "radiographic_findings": [
            "Homogeneous opacification of lung parenchyma",
            "Air bronchograms (air-filled bronchi visible within opacity)",
            "Silhouette sign (obliteration of adjacent structures)",
            "No significant volume loss",
            "Lobar or segmental distribution"
        ],
        "causes": [
            "Bacterial pneumonia (most common)",
            "Viral pneumonia",
            "Aspiration pneumonia",
            "Pulmonary hemorrhage",
            "Pulmonary edema",
            "Lung contusion",
            "Organizing pneumonia",
            "Malignancy (bronchioloalveolar carcinoma)"
        ],
        "risk_factors": [
            "Advanced age",
            "Immunocompromised state",
            "Chronic lung disease",
            "Smoking",
            "Recent viral infection",
            "Aspiration risk (dysphagia, altered consciousness)",
            "Healthcare exposure"
        ],
        "precautions": [
            "Appropriate antibiotic therapy if infectious",
            "Supportive care (oxygen, hydration)",
            "Monitor for respiratory deterioration",
            "Consider sputum culture and sensitivity",
            "Vaccination (pneumococcal, influenza)",
            "Follow-up imaging to ensure resolution",
            "Evaluate for underlying malignancy if persistent"
        ],
        "clinical_significance": "High",
        "urgency": "Prompt evaluation required"
    },
    
    "Edema": {
        "full_name": "Pulmonary Edema",
        "description": (
            "Pulmonary edema is the accumulation of fluid in the lungs, specifically "
            "in the alveoli and interstitial spaces. It can be cardiogenic (due to "
            "heart failure) or non-cardiogenic (ARDS, renal failure, fluid overload). "
            "This condition impairs gas exchange and can be life-threatening."
        ),
        "common_locations": [
            "Perihilar regions (bat wing pattern in cardiogenic)",
            "Bilateral and symmetric distribution",
            "Lower lung zones (gravity-dependent)",
            "Interlobular septa (Kerley B lines)",
            "Peribronchial cuffing"
        ],
        "radiographic_findings": [
            "Bilateral perihilar hazy opacities (bat wing appearance)",
            "Kerley B lines (interstitial edema)",
            "Peribronchial cuffing",
            "Cardiomegaly (if cardiogenic)",
            "Pleural effusions (often bilateral)",
            "Upper lobe vascular redistribution (cephalization)"
        ],
        "causes": [
            "Congestive heart failure (most common)",
            "Acute myocardial infarction",
            "Valvular heart disease",
            "Fluid overload",
            "Renal failure",
            "ARDS (acute respiratory distress syndrome)",
            "High altitude",
            "Drug toxicity"
        ],
        "risk_factors": [
            "Pre-existing heart disease",
            "Hypertension",
            "Coronary artery disease",
            "Chronic kidney disease",
            "Recent IV fluid administration",
            "Sepsis",
            "Trauma"
        ],
        "precautions": [
            "Immediate respiratory support if severe",
            "Diuretic therapy (furosemide)",
            "Position patient upright to reduce pulmonary congestion",
            "Oxygen therapy",
            "Treat underlying cause (cardiac vs non-cardiac)",
            "Fluid restriction",
            "Cardiac monitoring",
            "Consider intensive care admission if severe"
        ],
        "clinical_significance": "High",
        "urgency": "Urgent - may require immediate intervention"
    },
    
    "Effusion": {
        "full_name": "Pleural Effusion",
        "description": (
            "Pleural effusion is the abnormal accumulation of fluid in the pleural "
            "space, the area between the visceral and parietal pleura surrounding the "
            "lungs. It can be transudative (due to systemic factors) or exudative "
            "(due to local inflammation or malignancy)."
        ),
        "common_locations": [
            "Costophrenic angles (earliest finding)",
            "Lateral chest wall",
            "Posterior costophrenic sulcus (supine films)",
            "Subpulmonic space",
            "Fissures (loculated effusions)"
        ],
        "radiographic_findings": [
            "Blunting of costophrenic angle (minimum 200-300mL)",
            "Meniscus sign (curved upper border)",
            "Homogeneous opacity in dependent portions",
            "Mediastinal shift away from large effusions",
            "Lamellar effusion along lateral chest wall",
            "Subpulmonic effusion (elevated hemidiaphragm appearance)"
        ],
        "causes": [
            "Congestive heart failure (transudative)",
            "Pneumonia (parapneumonic effusion)",
            "Malignancy (lung cancer, metastases, mesothelioma)",
            "Pulmonary embolism",
            "Liver cirrhosis",
            "Nephrotic syndrome",
            "Tuberculosis",
            "Rheumatoid arthritis",
            "Trauma (hemothorax)"
        ],
        "risk_factors": [
            "Heart failure",
            "Liver disease",
            "Renal disease",
            "Known malignancy",
            "Recent infection",
            "Autoimmune conditions",
            "Recent surgery or trauma"
        ],
        "precautions": [
            "Diagnostic thoracentesis if cause unclear",
            "Analyze fluid (Light's criteria for transudate vs exudate)",
            "Treat underlying cause",
            "Therapeutic drainage if causing respiratory compromise",
            "Serial imaging to monitor progression",
            "Consider chest tube for large or complicated effusions",
            "Cytology if malignancy suspected",
            "Pleurodesis for recurrent malignant effusions"
        ],
        "clinical_significance": "Moderate to High",
        "urgency": "Depends on size and underlying cause"
    }
}


def get_disease_info(disease_name: str) -> dict:
    """
    Retrieve information about a specific disease.
    
    Args:
        disease_name: Name of the disease
        
    Returns:
        Dictionary containing disease information
    """
    # Handle case variations
    for key in DISEASE_INFO:
        if key.lower() == disease_name.lower():
            return DISEASE_INFO[key]
    
    return None


def get_disease_summary(disease_name: str) -> str:
    """
    Get a brief summary of disease information.
    
    Args:
        disease_name: Name of the disease
        
    Returns:
        Formatted summary string
    """
    info = get_disease_info(disease_name)
    
    if info is None:
        return f"No information available for {disease_name}"
    
    summary = []
    summary.append(f"\n{info['full_name']}")
    summary.append("=" * len(info['full_name']))
    summary.append(f"\n{info['description']}")
    
    summary.append("\n\nPrimary Causes:")
    for cause in info['causes'][:5]:
        summary.append(f"  - {cause}")
    
    summary.append("\n\nKey Precautions:")
    for precaution in info['precautions'][:5]:
        summary.append(f"  - {precaution}")
    
    summary.append(f"\n\nClinical Significance: {info['clinical_significance']}")
    summary.append(f"Urgency Level: {info['urgency']}")
    
    return "\n".join(summary)


def get_all_diseases() -> list:
    """Get list of all diseases with available information."""
    return list(DISEASE_INFO.keys())


def format_report_section(disease_name: str, probability: float, 
                          location: str = None) -> str:
    """
    Format a clinical report section for a detected disease.
    
    Args:
        disease_name: Name of the detected disease
        probability: Detection probability (0-1)
        location: Localization result from GradCAM analysis
        
    Returns:
        Formatted report section
    """
    info = get_disease_info(disease_name)
    
    if info is None:
        return f"Finding: {disease_name}\nConfidence: {probability:.1%}\n"
    
    report = []
    report.append(f"FINDING: {info['full_name']}")
    report.append(f"Detection Confidence: {probability:.1%}")
    
    if location:
        report.append(f"Approximate Location: {location}")
    
    report.append(f"\nDescription:\n{info['description']}")
    
    report.append("\nCommon Radiographic Findings:")
    for finding in info['radiographic_findings'][:4]:
        report.append(f"  - {finding}")
    
    report.append("\nPossible Causes:")
    for cause in info['causes'][:4]:
        report.append(f"  - {cause}")
    
    report.append("\nRecommended Actions:")
    for precaution in info['precautions'][:4]:
        report.append(f"  - {precaution}")
    
    report.append(f"\nClinical Significance: {info['clinical_significance']}")
    report.append(f"Urgency: {info['urgency']}")
    
    return "\n".join(report)
