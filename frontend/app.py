from flask import Flask, render_template, request, session, jsonify
from datetime import datetime
import uuid
import requests
import logging

app = Flask(__name__)
app.secret_key = 'temp_session_key_for_privacy'  # Only for temporary session, no data stored

# Configure logging to show in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # This will output to console
    ]
)

# Backend API configuration
BACKEND_API_URL = 'http://localhost:5001'
logger = logging.getLogger(__name__)

# PRIVACY NOTICE: This frontend application does NOT store any personal data
# All responses are sent to backend API for processing and discarded after results are shown

# AQ-20 Questions that match the training data
AQ20_QUESTIONS = [
    {
        "id": "A1_Score",
        "question": "I prefer to do things with others rather than on my own",
        "description": "Do you enjoy group activities more than solo activities?"
    },
    {
        "id": "A2_Score",
        "question": "I prefer to do things the same way over and over again",
        "description": "Do you like routine and predictable patterns?"
    },
    {
        "id": "A3_Score",
        "question": "If I try to imagine something, I find it very easy to create a picture in my mind",
        "description": "Can you easily visualize things in your mind?"
    },
    {
        "id": "A4_Score",
        "question": "I frequently get so strongly absorbed in one thing that I lose sight of other things",
        "description": "Do you get deeply focused on activities and lose track of surroundings?"
    },
    {
        "id": "A5_Score",
        "question": "I often notice small sounds when others do not",
        "description": "Do you notice subtle sounds that others might miss?"
    },
    {
        "id": "A6_Score",
        "question": "I usually notice car number plates or similar strings of information",
        "description": "Do you notice details like license plates or serial numbers?"
    },
    {
        "id": "A7_Score",
        "question": "Other people frequently tell me that what I've said is impolite, even though I think it is polite",
        "description": "Do others sometimes say you're being rude when you don't intend to be?"
    },
    {
        "id": "A8_Score",
        "question": "When I'm reading a story, I can easily imagine what the characters might look like",
        "description": "Can you easily picture characters when reading?"
    },
    {
        "id": "A9_Score",
        "question": "I am fascinated by dates",
        "description": "Do you find dates and calendars particularly interesting?"
    },
    {
        "id": "A10_Score",
        "question": "In a social group, I can easily keep track of several different people's conversations",
        "description": "Can you follow multiple conversations at once in groups?"
    },
    {
        "id": "A11_Score",
        "question": "I find social situations easy",
        "description": "Do you feel comfortable in social settings?"
    },
    {
        "id": "A12_Score",
        "question": "I tend to notice details that others do not",
        "description": "Do you often spot details that others miss?"
    },
    {
        "id": "A13_Score",
        "question": "I would rather go to a library than a party",
        "description": "Do you prefer quiet activities over social gatherings?"
    },
    {
        "id": "A14_Score",
        "question": "I find making up stories easy",
        "description": "Can you easily create fictional stories?"
    },
    {
        "id": "A15_Score",
        "question": "I find myself drawn more strongly to people than to things",
        "description": "Are you more interested in people than objects or systems?"
    },
    {
        "id": "A16_Score",
        "question": "I tend to have very strong interests which I get upset about if I can't pursue",
        "description": "Do you have intense interests that are important to you?"
    },
    {
        "id": "A17_Score",
        "question": "I enjoy social chit-chat",
        "description": "Do you like casual conversation with others?"
    },
    {
        "id": "A18_Score",
        "question": "When I talk, it isn't always easy for others to get a word in edgeways",
        "description": "Do you sometimes talk for long periods without pausing?"
    },
    {
        "id": "A19_Score",
        "question": "I am fascinated by numbers",
        "description": "Do you find numbers particularly interesting?"
    },
    {
        "id": "A20_Score",
        "question": "When I'm reading a story, I find it difficult to work out the characters' intentions",
        "description": "Do you have trouble understanding characters' motivations?"
    }
]

@app.route('/')
def landing():
    return render_template("landing.html")

@app.route('/index')
def quiz():
    """Load questions from backend API"""
    try:
        # Try to get questions from backend
        response = requests.get(f'{BACKEND_API_URL}/api/questions', timeout=5)
        if response.status_code == 200:
            api_data = response.json()
            if api_data.get('success'):
                questions = api_data.get('questions', [])
                logger.info(f"Loaded {len(questions)} questions from backend")
                return render_template("index.html", questions=questions, use_api=True)
    except requests.exceptions.RequestException as e:
        logger.warning(f"Backend API not available: {str(e)}")
    
    # Fallback to hardcoded AQ-20 questions if backend is not available
    return render_template("index.html", questions=AQ20_QUESTIONS, use_api=False)

@app.route('/result', methods=['POST'])
def results():
    print("🚀 RESULT ROUTE CALLED - Debug output enabled")
    logger.info("🚀 RESULT ROUTE CALLED - Starting assessment processing")
    
    # PRIVACY: Generate temporary session ID for this session only (not stored)
    if 'temp_session' not in session:
        session['temp_session'] = str(uuid.uuid4())[:8]  # Short ID for this session only
    
    # TEST: Quick backend health check first
    print("🔍 Testing backend health check...")
    try:
        health_response = requests.get(f'{BACKEND_API_URL}/api/health', timeout=5)
        print(f"🔍 Backend health check status: {health_response.status_code}")
        logger.info(f"🔍 Backend health check: {health_response.status_code}")
        if health_response.status_code == 200:
            print(f"✅ Backend is reachable at {BACKEND_API_URL}")
            logger.info(f"✅ Backend is reachable at {BACKEND_API_URL}")
        else:
            print(f"❌ Backend health check failed: {health_response.status_code}")
            logger.error(f"❌ Backend health check failed: {health_response.status_code}")
    except Exception as health_err:
        print(f"❌ Backend health check exception: {str(health_err)}")
        logger.error(f"❌ Backend health check exception: {str(health_err)}")
    
    try:
        # Prepare data for backend API
        assessment_data = {}
        
        # Collect AQ scores (try both AQ-10 and AQ-20)
        for i in range(1, 21):  # Support both AQ-10 and AQ-20
            field_name = f'A{i}_Score'
            if field_name in request.form:
                assessment_data[field_name] = int(request.form.get(field_name, 0))
        
        # DEBUG: Log the actual data being sent
        logger.info(f"🔍 DEBUG: AQ Scores being sent to backend:")
        for i in range(1, 21):
            field_name = f'A{i}_Score'
            score = assessment_data.get(field_name, 'MISSING')
            logger.info(f"  {field_name}: {score}")
        
        total_aq = sum(assessment_data.get(f'A{i}_Score', 0) for i in range(1, 21))
        logger.info(f"🔍 DEBUG: Total AQ Score calculated: {total_aq}")
        
        # Collect demographic data
        demographic_fields = ['age', 'gender', 'ethnicity', 'contry_of_res', 
                             'jaundice', 'austim', 'used_app_before', 'relation']
        
        for field in demographic_fields:
            if field in request.form and request.form[field]:
                assessment_data[field] = request.form[field]
        
        # Collect text responses if provided
        text_responses = {}
        text_fields = ['text_social', 'text_interests', 'text_daily', 'text_sensory']
        
        for field in text_fields:
            if field in request.form:
                text_value = request.form[field].strip()
                if text_value:  # Only include non-empty responses
                    text_responses[field] = text_value
        
        # Add text responses to assessment data if any exist
        if text_responses:
            assessment_data['text_responses'] = text_responses
            logger.info(f"📝 Including {len(text_responses)} text responses in analysis")
        
        logger.info(f"🔄 Sending assessment to backend API (session: {session['temp_session']})")
        logger.info(f"🔄 Backend URL: {BACKEND_API_URL}/api/predict")
        logger.info(f"🔄 Data being sent: {len(assessment_data)} fields")
        
        # Send to backend API for processing
        try:
            response = requests.post(
                f'{BACKEND_API_URL}/api/predict',
                json=assessment_data,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            logger.info(f"🔄 Backend response status: {response.status_code}")
            logger.info(f"🔄 Backend response: {response.text[:500]}...")  # First 500 chars
            
        except requests.exceptions.RequestException as req_err:
            logger.error(f"❌ Request failed: {str(req_err)}")
            raise req_err
        
        if response.status_code == 200:
            api_result = response.json()
            
            if api_result.get('success'):
                # Use API results - pass as 'results' object for template
                results_data = {
                    'prediction': api_result.get('prediction', 'Unknown'),
                    'confidence': api_result.get('confidence', 50),
                    'aq_score': api_result.get('aq_total_score', 0),
                    'risk_class': api_result.get('risk_level', 'unknown'),
                    'explanation': api_result.get('explanation', 'Assessment completed'),
                    'recommendations': api_result.get('recommendations', []),
                    'text_analysis': api_result.get('text_analysis'),  # Include text analysis
                    'session_id': session['temp_session'],
                    'privacy_notice': 'Your responses were processed privately and are not stored anywhere.',
                    'api_used': True,
                    'model_info': api_result.get('model_info', {})
                }
                
                logger.info(f"✅ Backend API prediction successful")
                return render_template('result.html', results=results_data)
            else:
                logger.error(f"Backend API error: {api_result.get('error', 'Unknown error')}")
                raise requests.exceptions.RequestException("API returned error")
        else:
            logger.error(f"Backend API HTTP error: {response.status_code}")
            raise requests.exceptions.RequestException(f"HTTP {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Backend API REQUEST FAILED: {str(e)}")
        logger.error(f"❌ Exception type: {type(e).__name__}")
        logger.error(f"❌ Full error details: {repr(e)}")
        
        # FALLBACK: Simple local processing if backend is unavailable
        logger.warning(f"🔄 USING FALLBACK LOCAL ASSESSMENT")
        aq_score = 0
        for i in range(1, 11):  # AQ-10 fallback
            score = request.form.get(f'A{i}_Score', 0)
            aq_score += int(score) if score else 0
        
        prediction = "ASD" if aq_score >= 6 else "No ASD"
        confidence = 70  # Lower confidence for fallback
        
        results_data = {
            'prediction': prediction,
            'confidence': confidence,
            'aq_score': aq_score,
            'risk_class': 'high' if aq_score >= 6 else 'low',
            'explanation': f'Your AQ-10 score is {aq_score}/10 (fallback mode)',
            'recommendations': [
                'Backend API unavailable - using simplified assessment',
                'For more accurate results, try again later',
                'Consult a healthcare professional for proper evaluation'
            ],
            'text_analysis': None,  # No text analysis in fallback mode
            'session_id': session['temp_session'],
            'privacy_notice': 'Your responses were processed privately and are not stored anywhere.',
            'api_used': False,
            'fallback_mode': True
        }
        
        return render_template('result.html', results=results_data)

@app.route('/api/status')
def api_status():
    """Check backend API status"""
    try:
        response = requests.get(f'{BACKEND_API_URL}/api/health', timeout=5)
        if response.status_code == 200:
            return jsonify({
                'backend_available': True,
                'backend_status': response.json()
            })
    except requests.exceptions.RequestException:
        pass
    
    return jsonify({
        'backend_available': False,
        'fallback_mode': True
    })

@app.route('/privacy')
def privacy_policy():
    """Privacy policy page emphasizing no data storage"""
    return render_template('privacy.html')

# Step 4: Run the app with privacy emphasis
if __name__ == '__main__':
    app.run(debug=True)
