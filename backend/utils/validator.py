"""
Form Validator - Validates user input for autism assessment
Privacy-preserving validation with no data logging
"""

import re
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class FormValidator:
    def __init__(self):
        self.required_aq_questions = [f'A{i}_Score' for i in range(1, 21)]  # AQ-20 questions
        self.required_demographics = ['age', 'gender', 'ethnicity', 'contry_of_res', 
                                    'jaundice', 'austim', 'used_app_before', 'relation']
        self.valid_responses = [0, 1, 2, 3]  # Accept 0-3 scale from frontend
        
    def validate_assessment_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete assessment data
        Returns: {valid: bool, errors: List[str], warnings: List[str]}
        """
        try:
            errors = []
            warnings = []
            
            if not data:
                return {
                    'valid': False,
                    'errors': ['No data provided'],
                    'warnings': []
                }
            
            # Validate AQ questions
            aq_errors, aq_warnings = self._validate_aq_questions(data)
            errors.extend(aq_errors)
            warnings.extend(aq_warnings)
            
            # Validate demographics
            demo_errors, demo_warnings = self._validate_demographics(data)
            errors.extend(demo_errors)
            warnings.extend(demo_warnings)
            
            # Overall validation
            is_valid = len(errors) == 0
            
            if is_valid:
                logger.info("✅ Form validation passed")
            else:
                logger.warning(f"❌ Form validation failed: {len(errors)} errors")
            
            return {
                'valid': is_valid,
                'errors': errors,
                'warnings': warnings,
                'total_questions_answered': self._count_answered_questions(data),
                'completion_percentage': self._calculate_completion_percentage(data)
            }
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return {
                'valid': False,
                'errors': ['Validation system error'],
                'warnings': []
            }
    
    def _validate_aq_questions(self, data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate AQ questionnaire responses"""
        errors = []
        warnings = []
        
        answered_questions = 0
        
        # Check each AQ question
        for question_id in self.required_aq_questions:
            if question_id in data:
                try:
                    response = int(data[question_id])
                    if response in self.valid_responses:
                        answered_questions += 1
                    else:
                        errors.append(f"Invalid response for {question_id}: must be 0 or 1")
                except (ValueError, TypeError):
                    errors.append(f"Invalid response type for {question_id}: must be a number")
            else:
                errors.append(f"Missing required question: {question_id}")
        
        # Check if minimum questions answered
        if answered_questions < len(self.required_aq_questions):
            errors.append(f"Only {answered_questions}/{len(self.required_aq_questions)} AQ questions answered")
        
        # Check for extended AQ-20 questions (optional)
        extended_questions = [f'A{i}_Score' for i in range(11, 21)]
        extended_answered = sum(1 for q in extended_questions if q in data and self._is_valid_aq_response(data[q]))
        
        if extended_answered > 0:
            warnings.append(f"Extended AQ-20 questions detected ({extended_answered}/10)")
        
        return errors, warnings
    
    def _validate_demographics(self, data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate demographic information"""
        errors = []
        warnings = []
        
        # Validate age
        if 'age' not in data:
            errors.append("Age is required")
        else:
            age_errors, age_warnings = self._validate_age(data['age'])
            errors.extend(age_errors)
            warnings.extend(age_warnings)
        
        # Validate gender
        if 'gender' not in data or not data['gender']:
            errors.append("Gender is required")
        else:
            gender_errors = self._validate_gender(data['gender'])
            errors.extend(gender_errors)
        
        # Validate ethnicity
        if 'ethnicity' not in data or not data['ethnicity']:
            warnings.append("Ethnicity not provided - using default")
        
        # Validate country
        if 'contry_of_res' not in data or not data['contry_of_res']:
            warnings.append("Country of residence not provided - using default")
        
        # Validate yes/no fields
        yes_no_fields = ['jaundice', 'austim', 'used_app_before']
        for field in yes_no_fields:
            if field not in data:
                warnings.append(f"{field} not provided - using default 'no'")
            else:
                if not self._is_valid_yes_no(data[field]):
                    warnings.append(f"Invalid {field} value - using default 'no'")
        
        # Validate relation
        if 'relation' not in data or not data['relation']:
            warnings.append("Relation not provided - using default 'Self'")
        else:
            relation_errors = self._validate_relation(data['relation'])
            errors.extend(relation_errors)
        
        return errors, warnings
    
    def _validate_age(self, age_value) -> Tuple[List[str], List[str]]:
        """Validate age field"""
        errors = []
        warnings = []
        
        try:
            age = int(float(age_value))
            if age < 1:
                errors.append("Age must be at least 1")
            elif age > 120:
                errors.append("Age must be less than 120")
            elif age < 12:
                warnings.append("Assessment designed for ages 12+")
            elif age > 80:
                warnings.append("Assessment primarily validated for younger adults")
        except (ValueError, TypeError):
            errors.append("Age must be a valid number")
        
        return errors, warnings
    
    def _validate_gender(self, gender_value) -> List[str]:
        """Validate gender field"""
        errors = []
        
        if not gender_value:
            errors.append("Gender cannot be empty")
            return errors
        
        valid_genders = ['m', 'f', 'male', 'female', 'other', 'prefer not to say']
        gender_lower = str(gender_value).lower().strip()
        
        if gender_lower not in valid_genders:
            # Not an error, just a warning - will be processed as 'other'
            pass
        
        return errors
    
    def _validate_relation(self, relation_value) -> List[str]:
        """Validate relation field"""
        errors = []
        
        if not relation_value:
            return errors  # Will use default
        
        valid_relations = ['self', 'parent', 'relative', 'health care professional', 'others']
        relation_lower = str(relation_value).lower().strip()
        
        if relation_lower not in valid_relations:
            # Not an error, will be processed as 'Others'
            pass
        
        return errors
    
    def _is_valid_aq_response(self, value) -> bool:
        """Check if AQ response is valid"""
        try:
            response = int(value)
            return response in self.valid_responses
        except (ValueError, TypeError):
            return False
    
    def _is_valid_yes_no(self, value) -> bool:
        """Check if yes/no response is valid"""
        if not value:
            return False
        
        value_lower = str(value).lower().strip()
        return value_lower in ['yes', 'no', 'y', 'n', 'true', 'false', '1', '0']
    
    def _count_answered_questions(self, data: Dict[str, Any]) -> int:
        """Count how many questions were answered"""
        count = 0
        
        # Count AQ questions
        for question_id in self.required_aq_questions:
            if question_id in data and self._is_valid_aq_response(data[question_id]):
                count += 1
        
        # Count demographics
        for field in self.required_demographics:
            if field in data and data[field] is not None and str(data[field]).strip():
                count += 1
        
        return count
    
    def _calculate_completion_percentage(self, data: Dict[str, Any]) -> float:
        """Calculate completion percentage"""
        total_fields = len(self.required_aq_questions) + len(self.required_demographics)
        answered_fields = self._count_answered_questions(data)
        
        if total_fields == 0:
            return 0.0
        
        return round((answered_fields / total_fields) * 100, 1)
    
    def validate_single_field(self, field_name: str, value: Any) -> Dict[str, Any]:
        """Validate a single field (for real-time validation)"""
        try:
            errors = []
            warnings = []
            
            if field_name in self.required_aq_questions:
                if not self._is_valid_aq_response(value):
                    errors.append(f"Invalid response for {field_name}")
            
            elif field_name == 'age':
                age_errors, age_warnings = self._validate_age(value)
                errors.extend(age_errors)
                warnings.extend(age_warnings)
            
            elif field_name == 'gender':
                gender_errors = self._validate_gender(value)
                errors.extend(gender_errors)
            
            elif field_name in ['jaundice', 'austim', 'used_app_before']:
                if not self._is_valid_yes_no(value):
                    warnings.append(f"Invalid yes/no value for {field_name}")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }
            
        except Exception as e:
            logger.error(f"Error validating field {field_name}: {str(e)}")
            return {
                'valid': False,
                'errors': [f"Validation error for {field_name}"],
                'warnings': []
            }
    
    def get_validation_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of validation status"""
        validation_result = self.validate_assessment_data(data)
        
        return {
            'is_valid': validation_result['valid'],
            'completion_percentage': validation_result['completion_percentage'],
            'total_errors': len(validation_result['errors']),
            'total_warnings': len(validation_result['warnings']),
            'ready_for_submission': validation_result['valid'] and validation_result['completion_percentage'] >= 90,
            'missing_required_fields': [error for error in validation_result['errors'] if 'Missing required' in error]
        }
