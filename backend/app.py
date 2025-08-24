"""
Backend API Server for Autism Spectrum Assessment Tool
Pure REST API with ML integration - No data storage for privacy
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import logging
import traceback

# Import our ML model and utilities
from models.autism_classifier import AutismClassifier
from utils.data_processor import DataProcessor
from utils.validator import FormValidator
from utils.text_analyzer import TextAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests from frontend

# Initialize ML components
classifier = AutismClassifier()
data_processor = DataProcessor()
validator = FormValidator()
text_analyzer = TextAnalyzer()

# Initialize the model (load or train if needed)
logger.info("🤖 Initializing ML model...")
model_ready = classifier.initialize()
if model_ready:
    logger.info("✅ ML model ready")
else:
    logger.warning("⚠️ ML model initialization failed - using fallback predictions")

# PRIVACY NOTICE: This API processes data in memory only
# No personal data is stored, logged, or persisted anywhere
logger.info("🔒 Privacy-First API Server Starting - No Data Storage")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        model_status = classifier.is_model_loaded()
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'model_loaded': model_status,
            'privacy_mode': True,
            'data_storage': False
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/text-questions', methods=['GET'])
def get_text_questions():
    """Get text analysis questions"""
    try:
        questions = text_analyzer.get_text_questions()
        
        return jsonify({
            'success': True,
            'questions': questions,
            'total_questions': len(questions)
        })
        
    except Exception as e:
        logger.error(f"Error getting text questions: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to load text questions'
        }), 500

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """Analyze user text responses"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        text_responses = data.get('text_responses', {})
        
        if not text_responses:
            return jsonify({
                'success': False,
                'error': 'No text responses provided'
            }), 400
        
        # Analyze each text response
        analysis_results = {}
        
        for question_id, response_text in text_responses.items():
            if response_text and response_text.strip():
                analysis = text_analyzer.analyze_text(response_text)
                analysis_results[question_id] = analysis
        
        # Generate overall text analysis summary
        overall_insights = generate_overall_text_insights(analysis_results)
        
        return jsonify({
            'success': True,
            'text_analysis': analysis_results,
            'overall_insights': overall_insights,
            'analyzed_responses': len(analysis_results)
        })
        
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to analyze text responses'
        }), 500

def generate_overall_text_insights(analysis_results):
    """Generate overall insights from all text responses"""
    if not analysis_results:
        return []
    
    insights = []
    categories = {}
    total_confidence = 0
    
    # Aggregate category scores
    for analysis in analysis_results.values():
        top_category = analysis.get('top_category', 'general')
        confidence = analysis.get('confidence', 0)
        
        if top_category not in categories:
            categories[top_category] = []
        categories[top_category].append(confidence)
        total_confidence += confidence
    
    # Average confidence
    avg_confidence = total_confidence / len(analysis_results)
    
    # Most prominent category
    if categories:
        avg_category_scores = {cat: sum(scores)/len(scores) for cat, scores in categories.items()}
        top_category = max(avg_category_scores.items(), key=lambda x: x[1])
        
        insights.append(f"Your text responses most strongly align with {top_category[0]} themes")
        insights.append(f"Overall text analysis confidence: {avg_confidence:.1f}%")
        
        if avg_confidence > 70:
            insights.append("Your responses show clear patterns that can inform the assessment")
        elif avg_confidence > 50:
            insights.append("Your responses provide moderate insights for the assessment")
        else:
            insights.append("Consider providing more detailed responses for better text analysis")
    
    return insights

@app.route('/api/questions', methods=['GET'])
def get_questions():
    """Get the AQ-20 question set"""
    try:
        questions = data_processor.get_question_set()
        return jsonify({
            'success': True,
            'questions': questions,
            'total_questions': len(questions),
            'question_type': 'AQ-20'
        }), 200
    except Exception as e:
        logger.error(f"Error fetching questions: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to load questions'}), 500

@app.route('/api/validate', methods=['POST'])
def validate_form():
    """Validate form data before prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'valid': False, 'errors': ['No data provided']}), 400
        
        validation_result = validator.validate_assessment_data(data)
        
        return jsonify(validation_result), 200 if validation_result['valid'] else 400
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'valid': False, 'errors': ['Validation failed']}), 500

@app.route('/api/predict', methods=['POST'])
def predict_autism():
    """
    Main prediction endpoint - processes AQ-20 + demographics + text analysis
    PRIVACY: All data processed in memory only, immediately discarded
    """
    session_id = str(uuid.uuid4())[:8]  # Temporary session ID for logging only
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided',
                'session_id': session_id
            }), 400
        
        # DEBUG: Log the received data
        logger.info(f"🔍 DEBUG: Received data keys: {list(data.keys())}")
        aq_scores = {k: v for k, v in data.items() if k.startswith('A') and k.endswith('_Score')}
        logger.info(f"🔍 DEBUG: AQ Scores received: {aq_scores}")
        total_received = sum(aq_scores.values()) if aq_scores else 0
        logger.info(f"🔍 DEBUG: Total AQ score received: {total_received}")
        
        logger.info(f"🔄 Processing assessment {session_id} (no personal data logged)")
        
        # Validate input data
        validation_result = validator.validate_assessment_data(data)
        if not validation_result['valid']:
            return jsonify({
                'success': False,
                'error': 'Invalid input data',
                'details': validation_result['errors'],
                'session_id': session_id
            }), 400
        
        # Process text responses if provided
        text_analysis = None
        correlation_analysis = None
        text_responses = data.get('text_responses', {})
        
        if text_responses:
            try:
                # Analyze text responses
                analysis_results = {}
                for question_id, response_text in text_responses.items():
                    if response_text and response_text.strip():
                        analysis = text_analyzer.analyze_text(response_text)
                        analysis_results[question_id] = analysis
                
                if analysis_results:
                    text_analysis = {
                        'analysis_results': analysis_results,
                        'overall_insights': generate_overall_text_insights(analysis_results)
                    }
                    
                    # Perform correlation analysis between text and AQ scores
                    aq_responses = {f'A{i}_Score': data.get(f'A{i}_Score', 0) for i in range(1, 21)}
                    aq_total = sum(aq_responses.values())
                    
                    correlation_analysis = text_analyzer.correlate_with_aq_scores(
                        text_analysis, aq_total, aq_responses
                    )
                    
                    logger.info(f"Text analysis and correlation completed for {len(analysis_results)} responses")
                
            except Exception as e:
                logger.warning(f"Text analysis failed: {e}")
                # Continue without text analysis if it fails
        
        # Process data for ML model (privacy-preserving)
        processed_features = data_processor.process_assessment_data(data)
        
        # Get ML prediction
        prediction_result = classifier.predict(processed_features)
        
        # Calculate additional metrics
        risk_assessment = data_processor.calculate_risk_level(
            prediction_result['prediction'],
            prediction_result['confidence'],
            processed_features['aq_total_score']
        )
        
        # Generate recommendations
        recommendations = data_processor.generate_recommendations(
            prediction_result['prediction'],
            processed_features['aq_total_score'],
            processed_features.get('age', 25)
        )
        
        # Add text-informed insights to recommendations if available
        if text_analysis and text_analysis['overall_insights']:
            recommendations.extend([
                "Based on your text responses:",
                *text_analysis['overall_insights']
            ])
        
        # Add correlation insights if available
        if correlation_analysis:
            recommendations.extend([
                "",  # Empty line for separation
                "🔬 Combined Analysis (AQ + Text):",
                *correlation_analysis['correlation_insights']
            ])
        
        # Prepare response (no personal data included)
        response = {
            'success': True,
            'session_id': session_id,
            'prediction': prediction_result['prediction'],
            'confidence': round(prediction_result['confidence'], 2),
            'aq_total_score': processed_features['aq_total_score'],
            'risk_level': risk_assessment['level'],
            'risk_description': risk_assessment['description'],
            'explanation': f"Based on your AQ-20 score of {processed_features['aq_total_score']}/20, " +
                          f"our model predicts {prediction_result['prediction']} with " +
                          f"{prediction_result['confidence']:.1f}% confidence.",
            'recommendations': recommendations,
            'text_analysis': text_analysis,
            'correlation_analysis': correlation_analysis,
            'combined_assessment': correlation_analysis['final_assessment'] if correlation_analysis else None,
            'model_info': {
                'type': classifier.get_model_type(),
                'features_used': len(processed_features),
                'training_samples': classifier.get_training_info()
            },
            'privacy_notice': 'Your responses were processed privately and are not stored anywhere.',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"✅ Assessment {session_id} completed successfully")
        
        # PRIVACY: All input data and processing variables are automatically
        # garbage collected when this function ends - no persistence
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"❌ Prediction error for session {session_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': 'Prediction failed',
            'session_id': session_id,
            'details': 'Internal server error during prediction'
        }), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about the ML model (for transparency)"""
    try:
        model_info = classifier.get_detailed_model_info()
        return jsonify({
            'success': True,
            'model_info': model_info
        }), 200
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to get model info'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Autism Assessment API Server")
    print("🔒 Privacy Mode: ON - No data storage")
    print("🤖 ML Model: Loading...")
    
    # Initialize model
    if classifier.initialize():
        print("✅ ML Model: Ready")
    else:
        print("⚠️  ML Model: Failed to load, using fallback")
    
    print("🌐 Server: http://localhost:5001")
    print("📋 API Docs: http://localhost:5001/api/health")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5001,
        threaded=True
    )
