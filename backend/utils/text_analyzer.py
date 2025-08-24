"""
Text Analysis Utility for Autism Spectrum Assessment
Analyzes user text responses using synthetic autism text data
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import logging

logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.reference_data = None
        self.reference_vectors = None
        self.load_reference_data()
        
    def load_reference_data(self):
        """Load and prepare the synthetic autism text data"""
        try:
            # Get the path to the data file - correct path from backend directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, '..', '..', 'data', 'synthetic_autism_text_data.csv')
            
            # Load the data
            self.reference_data = pd.read_csv(data_path)
            logger.info(f"Loaded {len(self.reference_data)} reference text samples")
            
            # Prepare TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2)
            )
            
            # Fit vectorizer on reference texts
            reference_texts = self.reference_data['text'].fillna('').astype(str)
            self.reference_vectors = self.vectorizer.fit_transform(reference_texts)
            
            logger.info("Text analysis model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            self.reference_data = None
            self.reference_vectors = None
    
    def analyze_text(self, user_text):
        """
        Analyze user text and provide insights
        
        Args:
            user_text (str): User's text response
            
        Returns:
            dict: Analysis results with similarity scores and insights
        """
        if not user_text or not user_text.strip():
            return {
                'similarity_scores': {},
                'top_category': 'general',
                'confidence': 0.0,
                'insights': ['No text provided for analysis'],
                'text_features': {
                    'word_count': 0,
                    'length': 0,
                    'complexity': 'low'
                }
            }
        
        if self.reference_data is None or self.vectorizer is None:
            return {
                'similarity_scores': {},
                'top_category': 'general',
                'confidence': 0.0,
                'insights': ['Text analysis model not available'],
                'text_features': {
                    'word_count': len(user_text.split()) if user_text else 0,
                    'length': len(user_text) if user_text else 0,
                    'complexity': 'unknown'
                }
            }
        
        try:
            # Clean and prepare user text
            cleaned_text = self.clean_text(user_text)
            
            # Extract basic text features
            text_features = self.extract_text_features(cleaned_text)
            
            # Transform user text using fitted vectorizer
            user_vector = self.vectorizer.transform([cleaned_text])
            
            # Calculate similarities with reference categories
            similarity_scores = self.calculate_category_similarities(user_vector)
            
            # Get insights based on analysis
            insights = self.generate_insights(similarity_scores, text_features)
            
            # Determine top category and confidence
            if similarity_scores:
                top_category = max(similarity_scores.items(), key=lambda x: x[1])
            else:
                top_category = ('general', 0.0)
            
            return {
                'similarity_scores': similarity_scores,
                'top_category': top_category[0],
                'confidence': round(top_category[1] * 100, 1),
                'insights': insights,
                'text_features': text_features
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {
                'similarity_scores': {},
                'top_category': 'general',
                'confidence': 0.0,
                'insights': ['Text analysis error - using basic assessment'],
                'text_features': {
                    'word_count': len(user_text.split()) if user_text else 0,
                    'length': len(user_text) if user_text else 0,
                    'complexity': 'unknown'
                }
            }
    
    def clean_text(self, text):
        """Clean and normalize text for analysis"""
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()
    
    def extract_text_features(self, text):
        """Extract basic features from text"""
        if not text:
            return {
                'word_count': 0,
                'length': 0,
                'complexity': 'low'
            }
        
        words = text.split()
        word_count = len(words)
        text_length = len(text)
        
        # Simple complexity measure based on average word length
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        
        if avg_word_length > 6:
            complexity = 'high'
        elif avg_word_length > 4:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        return {
            'word_count': word_count,
            'length': text_length,
            'complexity': complexity,
            'avg_word_length': round(avg_word_length, 1)
        }
    
    def calculate_category_similarities(self, user_vector):
        """Calculate similarity between user text and each category"""
        category_similarities = {}
        
        try:
            # Get unique categories
            categories = self.reference_data['category'].unique()
            
            for category in categories:
                # Get vectors for this category
                category_mask = self.reference_data['category'] == category
                category_vectors = self.reference_vectors[category_mask]
                
                # Calculate cosine similarities
                similarities = cosine_similarity(user_vector, category_vectors)
                
                # Handle the result properly - convert to numpy array if needed
                if hasattr(similarities, 'flatten'):
                    similarities = similarities.flatten()
                else:
                    similarities = np.array(similarities).flatten()
                
                # Use maximum similarity as category score
                if len(similarities) > 0:
                    category_similarities[category] = float(np.max(similarities))
                else:
                    category_similarities[category] = 0.0
            
            return category_similarities
            
        except Exception as e:
            logger.error(f"Error in category similarity calculation: {e}")
            # Return default similarities
            return {
                'social': 0.3,
                'sensory': 0.3,
                'interests': 0.3,
                'routine': 0.3,
                'emotional': 0.3
            }
    
    def generate_insights(self, similarity_scores, text_features):
        """Generate insights based on analysis results"""
        insights = []
        
        # Get top categories
        sorted_categories = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        top_3_categories = sorted_categories[:3]
        
        # Insight based on top category
        top_category, top_score = top_3_categories[0]
        
        if top_score > 0.7:
            insights.append(f"🔍 Your response shows strong alignment with {top_category} expressions commonly found in autism spectrum experiences")
        elif top_score > 0.5:
            insights.append(f"🔍 Your response shows moderate alignment with {top_category} themes in autism spectrum expressions")
        else:
            insights.append("🔍 Your response shows a unique perspective that doesn't strongly align with typical autism spectrum expression patterns")
        
        # Insights based on text features
        word_count = text_features['word_count']
        complexity = text_features['complexity']
        
        if word_count < 5:
            insights.append("💭 Consider providing more detail in your response for better analysis")
        elif word_count > 50:
            insights.append("💭 Your detailed response provides rich information for analysis")
        
        if complexity == 'high':
            insights.append("🧠 Your use of complex language suggests thoughtful self-reflection")
        
        # Category-specific insights with autism indicators
        if top_category == 'social':
            insights.append("👥 Your response focuses on social interactions - a key area in autism assessment")
            if top_score > 0.6:
                insights.append("📋 This aligns with common autism traits related to social communication differences")
        elif top_category == 'sensory':
            insights.append("🌟 Your response highlights sensory processing experiences - highly relevant to autism")
            if top_score > 0.6:
                insights.append("📋 Sensory sensitivity is a core feature of autism spectrum conditions")
        elif top_category == 'interests':
            insights.append("🎯 Your response reflects focused interests and passionate engagement")
            if top_score > 0.6:
                insights.append("📋 Intense, specific interests are a common autism spectrum trait")
        elif top_category == 'routine':
            insights.append("📅 Your response emphasizes structure and predictability preferences")
            if top_score > 0.6:
                insights.append("📋 Need for routine and predictability is typical in autism spectrum conditions")
        elif top_category == 'emotional':
            insights.append("💝 Your response shows awareness of emotional processing differences")
            if top_score > 0.6:
                insights.append("📋 Different emotional processing patterns are common in autism")
        
        # Add overall assessment insight
        high_scores = [score for _, score in sorted_categories if score > 0.6]
        if len(high_scores) >= 2:
            insights.append("🔬 Multiple high-similarity scores suggest strong alignment with autism spectrum expression patterns")
        elif len(high_scores) == 1:
            insights.append("🔬 One strong alignment area identified - consistent with some autism spectrum traits")
        else:
            insights.append("🔬 Lower similarity scores suggest less typical autism spectrum expression patterns")
        
        return insights
    
    def get_text_questions(self):
        """Get appropriate questions for text analysis based on the data"""
        questions = [
            {
                'id': 'text_social',
                'category': 'social',
                'question': 'Describe how you feel in social situations or group conversations. What aspects are easy or challenging for you?',
                'placeholder': 'For example: I find it easier to talk one-on-one than in groups...'
            },
            {
                'id': 'text_interests',
                'category': 'interests',
                'question': 'Tell us about something you\'re really interested in or passionate about. How do you engage with this interest?',
                'placeholder': 'For example: I love learning everything about trains and can talk about them for hours...'
            },
            {
                'id': 'text_daily',
                'category': 'daily_life',
                'question': 'Describe your ideal daily routine or how you prefer to organize your day. What helps you function best?',
                'placeholder': 'For example: I work better when I know what to expect and have a clear schedule...'
            },
            {
                'id': 'text_sensory',
                'category': 'sensory',
                'question': 'Describe any sounds, textures, lights, or other sensory experiences that affect you strongly (positively or negatively).',
                'placeholder': 'For example: Loud sudden noises make me feel overwhelmed, but I love soft textures...'
            }
        ]
        
        return questions

    def correlate_with_aq_scores(self, text_analysis_results, aq_score, aq_responses):
        """
        Correlate text analysis results with AQ questionnaire scores
        
        Args:
            text_analysis_results (dict): Results from text analysis
            aq_score (int): Total AQ-20 score
            aq_responses (dict): Individual AQ responses
            
        Returns:
            dict: Correlation analysis and combined insights
        """
        correlation_insights = []
        overall_consistency = "unknown"
        
        try:
            # Analyze consistency between text and AQ scores
            text_categories = text_analysis_results.get('analysis_results', {})
            high_text_scores = []
            
            # Identify categories with high text similarity
            for question_id, analysis in text_categories.items():
                if analysis.get('confidence', 0) > 60:
                    high_text_scores.append(analysis.get('top_category', ''))
            
            # AQ score interpretation
            if aq_score >= 12:
                aq_interpretation = "high"
                correlation_insights.append("🎯 Your AQ-20 score suggests significant autism spectrum traits")
            elif aq_score >= 8:
                aq_interpretation = "moderate"
                correlation_insights.append("🎯 Your AQ-20 score shows moderate autism spectrum indicators")
            else:
                aq_interpretation = "low"
                correlation_insights.append("🎯 Your AQ-20 score is below typical autism spectrum thresholds")
            
            # Correlate text themes with AQ domains
            social_aq_questions = [1, 6, 10, 11, 15]  # Social communication questions
            social_aq_score = sum(aq_responses.get(f'A{i}_Score', 0) for i in social_aq_questions)
            
            routine_aq_questions = [2, 4, 5, 9, 16]  # Routine/flexibility questions  
            routine_aq_score = sum(aq_responses.get(f'A{i}_Score', 0) for i in routine_aq_questions)
            
            attention_aq_questions = [3, 7, 8, 12, 13]  # Attention to detail questions
            attention_aq_score = sum(aq_responses.get(f'A{i}_Score', 0) for i in attention_aq_questions)
            
            # Check for consistency
            consistency_checks = []
            
            # Social consistency
            has_social_text = any('social' in cat for cat in high_text_scores)
            high_social_aq = social_aq_score >= 8  # High social difficulties
            
            if has_social_text and high_social_aq:
                consistency_checks.append("social_consistent")
                correlation_insights.append("✅ Your text responses and AQ scores both indicate social communication differences")
            elif has_social_text and not high_social_aq:
                consistency_checks.append("social_text_higher")
                correlation_insights.append("🔍 Your text responses suggest more social awareness than your AQ scores indicate")
            elif not has_social_text and high_social_aq:
                consistency_checks.append("social_aq_higher")
                correlation_insights.append("🔍 Your AQ scores suggest social difficulties not reflected in your text responses")
            
            # Sensory consistency
            has_sensory_text = any('sensory' in cat for cat in high_text_scores)
            if has_sensory_text:
                correlation_insights.append("✅ Your text responses indicate sensory processing differences - a key autism indicator")
            
            # Interest consistency
            has_interest_text = any('interest' in cat for cat in high_text_scores)
            if has_interest_text and attention_aq_score >= 8:
                consistency_checks.append("interest_consistent")
                correlation_insights.append("✅ Both your text and AQ responses suggest intense, focused interests")
            
            # Overall consistency assessment
            if len(consistency_checks) >= 2:
                overall_consistency = "high"
                correlation_insights.append("🎯 HIGH CONSISTENCY: Your text responses and AQ scores show strong alignment")
            elif len(consistency_checks) == 1:
                overall_consistency = "moderate"
                correlation_insights.append("🎯 MODERATE CONSISTENCY: Some alignment between text and AQ responses")
            else:
                overall_consistency = "low"
                correlation_insights.append("🎯 MIXED SIGNALS: Your text and AQ responses show different patterns")
            
            # Generate final assessment
            final_assessment = self._generate_combined_assessment(
                aq_interpretation, overall_consistency, high_text_scores, aq_score
            )
            
            return {
                'correlation_insights': correlation_insights,
                'overall_consistency': overall_consistency,
                'aq_interpretation': aq_interpretation,
                'text_themes': high_text_scores,
                'final_assessment': final_assessment,
                'domain_scores': {
                    'social_communication': social_aq_score,
                    'routine_flexibility': routine_aq_score,
                    'attention_detail': attention_aq_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {
                'correlation_insights': ["Unable to perform correlation analysis"],
                'overall_consistency': "unknown",
                'aq_interpretation': "unknown",
                'text_themes': [],
                'final_assessment': "Assessment data incomplete",
                'domain_scores': {}
            }
    
    def _generate_combined_assessment(self, aq_interpretation, consistency, text_themes, aq_score):
        """Generate a combined assessment based on all factors"""
        
        if aq_interpretation == "high" and consistency == "high":
            return f"🔴 STRONG INDICATORS: Both your AQ-20 score ({aq_score}/20) and text responses strongly suggest autism spectrum traits. Consider professional evaluation."
        
        elif aq_interpretation == "high" and consistency == "moderate":
            return f"🟡 MODERATE-HIGH INDICATORS: Your AQ-20 score ({aq_score}/20) is high with some supporting text evidence. Professional consultation recommended."
        
        elif aq_interpretation == "moderate" and consistency == "high":
            return f"🟡 MODERATE INDICATORS: Your AQ-20 score ({aq_score}/20) is moderate but text responses show clear autism-related patterns."
        
        elif aq_interpretation == "moderate" and consistency == "moderate":
            return f"🟡 MIXED INDICATORS: Your AQ-20 score ({aq_score}/20) and text responses show some autism spectrum patterns worth exploring."
        
        elif aq_interpretation == "low" and text_themes:
            return f"🟢 LOW AQ WITH TEXT INSIGHTS: Your AQ-20 score ({aq_score}/20) is low, but text responses reveal some relevant experiences."
        
        elif aq_interpretation == "low" and consistency == "low":
            return f"🟢 LOW INDICATORS: Your AQ-20 score ({aq_score}/20) and text responses suggest fewer autism spectrum traits."
        
        else:
            return f"📊 MIXED RESULTS: Your assessment shows varied patterns (AQ: {aq_score}/20). Consider discussing with a professional for clarity."
