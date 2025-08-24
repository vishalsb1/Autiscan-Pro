"""
Data Processor - Handles data processing for autism assessment
Privacy-preserving processing with no data persistence
"""

import json
import os
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.questions_path = "../data/questions.json"
        self.aq_questions = None
        self.load_questions()
    
    def load_questions(self):
        """Load AQ-20 questions from JSON file"""
        try:
            if os.path.exists(self.questions_path):
                with open(self.questions_path, 'r', encoding='utf-8') as f:
                    self.aq_questions = json.load(f)
                logger.info(f"Loaded {len(self.aq_questions)} AQ questions")
            else:
                logger.warning(f"Questions file not found at {self.questions_path}")
                # Fallback to AQ-10 questions
                self.aq_questions = self.get_fallback_aq10_questions()
        except Exception as e:
            logger.error(f"Error loading questions: {str(e)}")
            self.aq_questions = self.get_fallback_aq10_questions()
    
    def get_fallback_aq10_questions(self):
        """Fallback AQ-10 questions if JSON file is not available"""
        return {
            "A1_Score": "I often notice small sounds when others do not",
            "A2_Score": "I usually concentrate more on the whole picture, rather than the small details",
            "A3_Score": "I find it easy to do more than one thing at once",
            "A4_Score": "If there is an interruption, I can switch back to what I was doing very quickly",
            "A5_Score": "I find it easy to 'read between the lines' when someone is talking to me",
            "A6_Score": "I know how to tell if someone listening to me is getting bored",
            "A7_Score": "When I'm reading a story I find it difficult to work out the characters' intentions",
            "A8_Score": "I like to collect information about categories of things",
            "A9_Score": "I find it easy to work out what someone is thinking or feeling by looking at their face",
            "A10_Score": "I find it difficult to work out people's intentions"
        }
    
    def get_question_set(self):
        """Get the question set with additional metadata"""
        if not self.aq_questions:
            return []
        
        questions = []
        for q_id, question_text in self.aq_questions.items():
            questions.append({
                "id": q_id,
                "question": question_text,
                "type": "likert_4",
                "options": [
                    {"value": 1, "label": "Definitely Agree"},
                    {"value": 1, "label": "Slightly Agree"},
                    {"value": 0, "label": "Slightly Disagree"},
                    {"value": 0, "label": "Definitely Disagree"}
                ]
            })
        
        return questions
    
    def process_assessment_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw assessment data into features for ML model
        PRIVACY: Data processed in memory only, no persistence
        """
        try:
            processed = {}
            
            # Process AQ scores
            aq_scores = {}
            aq_total = 0
            
            # Process AQ-20 questions (or available subset)
            # Convert 0-3 scale to 0-1 binary scale that the model expects
            for i in range(1, 21):  # AQ-20
                field_name = f'A{i}_Score'
                if field_name in raw_data:
                    raw_score = int(raw_data[field_name]) if raw_data[field_name] else 0
                    # Convert 0-3 scale to 0-1 binary: 0,1 -> 0 and 2,3 -> 1
                    binary_score = 1 if raw_score >= 2 else 0
                    aq_scores[field_name] = binary_score
                    aq_total += binary_score
                    logger.debug(f"🔄 {field_name}: {raw_score} -> {binary_score}")
                else:
                    aq_scores[field_name] = 0
            
            processed.update(aq_scores)
            processed['aq_total_score'] = aq_total
            
            logger.info(f"🔄 Processed assessment data: AQ total = {aq_total} (converted from 0-3 to 0-1 scale)")
            
            # Process demographics
            demographics = {
                'age': self._process_age(raw_data.get('age')),
                'gender': self._process_gender(raw_data.get('gender')),
                'ethnicity': self._process_ethnicity(raw_data.get('ethnicity')),
                'jaundice': self._process_yes_no(raw_data.get('jaundice')),
                'austim': self._process_yes_no(raw_data.get('austim')),  # family history
                'contry_of_res': self._process_country(raw_data.get('contry_of_res')),
                'used_app_before': self._process_yes_no(raw_data.get('used_app_before')),
                'relation': self._process_relation(raw_data.get('relation'))
            }
            
            processed.update(demographics)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing assessment data: {str(e)}")
            # Return minimal valid data structure
            return {
                'aq_total_score': 0,
                'age': 25,
                'gender': 'unknown',
                'ethnicity': 'unknown',
                'jaundice': 'no',
                'austim': 'no',
                'contry_of_res': 'unknown',
                'used_app_before': 'no',
                'relation': 'Self'
            }
    
    def _process_age(self, age_value):
        """Process age field"""
        try:
            if age_value:
                age = int(float(age_value))
                # Reasonable bounds for age
                if 1 <= age <= 120:
                    return age
            return 25  # Default age
        except (ValueError, TypeError):
            return 25
    
    def _process_gender(self, gender_value):
        """Process gender field"""
        if not gender_value:
            return 'unknown'
        
        gender = str(gender_value).lower().strip()
        if gender in ['m', 'male', 'man']:
            return 'm'
        elif gender in ['f', 'female', 'woman']:
            return 'f'
        else:
            return 'unknown'
    
    def _process_ethnicity(self, ethnicity_value):
        """Process ethnicity field"""
        if not ethnicity_value:
            return 'unknown'
        
        ethnicity = str(ethnicity_value).strip()
        # Standardize common ethnicity values
        ethnicity_mapping = {
            'white-european': 'White-European',
            'white european': 'White-European',
            'asian': 'Asian',
            'black': 'Black',
            'hispanic': 'Hispanic',
            'latino': 'Latino',
            'middle eastern': 'Middle Eastern',
            'south asian': 'South Asian',
            'turkish': 'Turkish',
            'native indian': 'Native Indian',
            'pacifica': 'Pacifica',
            'pasifika': 'Pasifika'
        }
        
        return ethnicity_mapping.get(ethnicity.lower(), ethnicity)
    
    def _process_yes_no(self, value):
        """Process yes/no fields"""
        if not value:
            return 'no'
        
        value_lower = str(value).lower().strip()
        if value_lower in ['yes', 'y', 'true', '1']:
            return 'yes'
        else:
            return 'no'
    
    def _process_country(self, country_value):
        """Process country field"""
        if not country_value:
            return 'unknown'
        
        country = str(country_value).strip()
        # Standardize common country names
        country_mapping = {
            'usa': 'United States',
            'us': 'United States',
            'uk': 'United Kingdom',
            'britain': 'United Kingdom',
            'uae': 'United Arab Emirates'
        }
        
        return country_mapping.get(country.lower(), country)
    
    def _process_relation(self, relation_value):
        """Process relation field"""
        if not relation_value:
            return 'Self'
        
        relation = str(relation_value).strip()
        valid_relations = ['Self', 'Parent', 'Relative', 'Health care professional', 'Others']
        
        # Case-insensitive matching
        for valid_rel in valid_relations:
            if relation.lower() == valid_rel.lower():
                return valid_rel
        
        return 'Others'
    
    def calculate_risk_level(self, prediction: str, confidence: float, aq_score: int) -> Dict[str, str]:
        """Calculate risk level based on prediction and scores"""
        try:
            if prediction == "ASD":
                if confidence >= 80:
                    level = "high"
                    description = "High likelihood of autism spectrum traits"
                elif confidence >= 60:
                    level = "medium"
                    description = "Moderate likelihood of autism spectrum traits"
                else:
                    level = "low-medium"
                    description = "Some indication of autism spectrum traits"
            else:
                if confidence >= 80:
                    level = "low"
                    description = "Low likelihood of autism spectrum traits"
                elif confidence >= 60:
                    level = "low-medium"
                    description = "Minimal indication of autism spectrum traits"
                else:
                    level = "medium"
                    description = "Uncertain - consider professional evaluation"
            
            return {
                "level": level,
                "description": description
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk level: {str(e)}")
            return {
                "level": "unknown",
                "description": "Unable to determine risk level"
            }
    
    def generate_recommendations(self, prediction: str, aq_score: int, age: int) -> List[str]:
        """Generate personalized recommendations based on results"""
        try:
            recommendations = []
            
            if prediction == "ASD":
                recommendations.extend([
                    "Consider consulting with a qualified healthcare professional who specializes in autism spectrum disorders",
                    "A comprehensive evaluation by a psychologist or psychiatrist can provide a formal diagnosis",
                    "Explore autism support groups and communities that can provide valuable resources",
                    "Learn about autism spectrum conditions to better understand your experiences"
                ])
                
                if age < 18:
                    recommendations.append("Discuss these results with your parents or guardians")
                elif age >= 65:
                    recommendations.append("Consider discussing with your primary care physician first")
                
                if aq_score >= 15:
                    recommendations.append("Your high AQ score suggests seeking evaluation sooner rather than later")
                
            else:
                recommendations.extend([
                    "Your results suggest lower likelihood of autism spectrum traits",
                    "If you continue to have concerns, consider speaking with a healthcare professional",
                    "Remember that this is a screening tool, not a diagnostic assessment",
                    "Stay informed about neurodiversity and be supportive of others who may be on the autism spectrum"
                ])
                
                if aq_score >= 4:
                    recommendations.append("While your overall score is lower, some traits were identified - professional consultation may still be helpful")
            
            # Universal recommendations
            recommendations.extend([
                "This screening provides initial insights only - professional evaluation is needed for formal assessment",
                "Take care of your mental health and well-being regardless of the results",
                "Every person has unique strengths and challenges - celebrate your individuality"
            ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return [
                "Consult with a qualified healthcare professional for proper evaluation",
                "This screening tool provides initial insights only",
                "Professional medical evaluation is recommended for any concerns"
            ]
