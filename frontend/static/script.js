// Autism Assessment Tool - Frontend JavaScript
// Updated for Backend API Integration
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('assessmentForm');
    const submitBtn = document.getElementById('submitBtn');
    const resultsSection = document.getElementById('resultsSection');
    const loader = document.querySelector('.btn-loader');
    
    // Check if we have API or fallback questions
    const useApi = window.useApiQuestions || false;
    
    // Progress tracking - detect AQ-10 vs AQ-20
    let totalQuestions = detectQuestionCount();
    let totalFields = 8; // Age, gender, ethnicity, country, jaundice, autism, app, relation
    let totalFormFields = totalQuestions + totalFields;
    
    console.log(`🔍 Detected ${totalQuestions} AQ questions, using ${useApi ? 'API' : 'fallback'} mode`);
    
    // Initialize form validation
    initializeFormValidation();
    
    // Check backend API status
    checkBackendStatus();
    
    function detectQuestionCount() {
        // Count actual AQ questions in the form
        const aqInputs = form.querySelectorAll('input[name*="_Score"]');
        const uniqueQuestions = new Set();
        
        aqInputs.forEach(input => {
            const match = input.name.match(/A(\d+)_Score/);
            if (match) {
                uniqueQuestions.add(parseInt(match[1]));
            }
        });
        
        return uniqueQuestions.size || 10; // Default to 10 if detection fails
    }
    
    function checkBackendStatus() {
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                const statusIndicator = document.getElementById('api-status');
                if (statusIndicator) {
                    if (data.backend_available) {
                        statusIndicator.innerHTML = '🟢 Advanced ML Model Active';
                        statusIndicator.className = 'api-status online';
                    } else {
                        statusIndicator.innerHTML = '🟡 Fallback Mode Active';
                        statusIndicator.className = 'api-status fallback';
                    }
                }
            })
            .catch(error => {
                console.log('Backend status check failed:', error);
            });
    }
    
    function initializeFormValidation() {
        // Add event listeners to all form inputs
        const inputs = form.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            input.addEventListener('change', updateFormProgress);
            input.addEventListener('input', updateFormProgress);
        });
        
        // Initialize character counting for text areas
        initializeCharacterCounting();
        
        // Initial validation check
        updateFormProgress();
    }
    
    function initializeCharacterCounting() {
        const textareas = form.querySelectorAll('textarea');
        textareas.forEach(textarea => {
            const maxLength = parseInt(textarea.getAttribute('maxlength')) || 500;
            const charCountEl = textarea.parentElement.querySelector('.char-count .current-count');
            
            function updateCharCount() {
                const currentLength = textarea.value.length;
                if (charCountEl) {
                    charCountEl.textContent = currentLength;
                    
                    const charCountContainer = charCountEl.parentElement;
                    charCountContainer.classList.remove('warning', 'limit');
                    
                    if (currentLength > maxLength * 0.9) {
                        charCountContainer.classList.add('limit');
                    } else if (currentLength > maxLength * 0.75) {
                        charCountContainer.classList.add('warning');
                    }
                }
            }
            
            textarea.addEventListener('input', updateCharCount);
            updateCharCount(); // Initial count
        });
    }
    
    function updateFormProgress() {
        const filledFields = countFilledFields();
        const requiredFieldsComplete = countRequiredFields();
        const progress = (filledFields / totalFormFields) * 100;
        
        // Update progress bar if exists
        const progressBar = document.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.style.width = progress + '%';
        }
        
        // Update submit button state based on REQUIRED fields only
        updateSubmitButton(requiredFieldsComplete);
        
        // Update progress text if exists
        const progressText = document.querySelector('.progress-text');
        if (progressText) {
            const requiredFieldsText = requiredFieldsComplete ? "✓ All required fields completed" : "Required fields remaining";
            progressText.textContent = `${filledFields}/${totalFormFields} fields filled • ${requiredFieldsText}`;
        }
    }
    
    function countFilledFields() {
        let count = 0;
        
        // Count AQ questions (radio buttons)
        for (let i = 1; i <= 20; i++) { // Check up to AQ-20
            const questionName = `A${i}_Score`;
            const selectedOption = form.querySelector(`input[name="${questionName}"]:checked`);
            if (selectedOption) {
                count++;
            }
        }
        
        // Count demographic fields
        const demographicFields = ['age', 'gender', 'ethnicity', 'contry_of_res', 'jaundice', 'austim', 'used_app_before', 'relation'];
        demographicFields.forEach(fieldName => {
            const field = form.querySelector(`[name="${fieldName}"]`);
            if (field && field.value.trim() !== '') {
                count++;
            }
        });
        
        return count;
    }
    
    function updateSubmitButton(isFormComplete) {
        const allRequiredFieldsFilled = countRequiredFields();
        const enableSubmit = allRequiredFieldsFilled; // All required fields must be filled
        
        if (enableSubmit) {
            submitBtn.disabled = false;
            submitBtn.classList.remove('disabled');
            submitBtn.style.opacity = '1';
        } else {
            submitBtn.disabled = true;
            submitBtn.classList.add('disabled');
            submitBtn.style.opacity = '0.6';
        }
    }
    
    function countRequiredFields() {
        let allRequiredFilled = true;
        
        // Check AQ questions (all required)
        for (let i = 1; i <= 20; i++) {
            const questionName = `A${i}_Score`;
            const selectedOption = form.querySelector(`input[name="${questionName}"]:checked`);
            if (!selectedOption) {
                allRequiredFilled = false;
                break;
            }
        }
        
        // Check required demographic fields
        const requiredFields = ['age', 'gender', 'ethnicity', 'contry_of_res', 'jaundice', 'austim', 'used_app_before', 'relation'];
        requiredFields.forEach(fieldName => {
            const field = form.querySelector(`[name="${fieldName}"][required]`);
            if (field && (!field.value || field.value.trim() === '')) {
                allRequiredFilled = false;
            }
        });
        
        return allRequiredFilled;
    }
    
    // Form is now submitted normally to Flask route (which calls backend API)
    // No need for custom AJAX handling since Flask handles the API communication
    
    function showLoadingState() {
        submitBtn.disabled = true;
        const btnText = submitBtn.querySelector('span');
        if (btnText) btnText.style.display = 'none';
        if (loader) loader.classList.remove('hidden');
    }
    
    function hideLoadingState() {
        submitBtn.disabled = false;
        const btnText = submitBtn.querySelector('span');
        if (btnText) btnText.style.display = 'inline';
        if (loader) loader.classList.add('hidden');
    }
    
    // Add form submission handler for loading states
    if (form) {
        form.addEventListener('submit', function(e) {
            showLoadingState();
            
            // Add a small delay to show loading state
            setTimeout(() => {
                // Form will submit normally to Flask
            }, 100);
        });
    }
    
    // Global functions for buttons (if results page needs them)
    window.resetAssessment = function() {
        // Reset form if we're on the assessment page
        if (form) {
            form.reset();
            updateFormProgress();
        }
        
        // Navigate back to assessment
        window.location.href = '/index';
    };
    
    window.printResults = function() {
        window.print();
    };
    
    window.checkApiStatus = function() {
        checkBackendStatus();
    };
    
    // Initialize the form
    console.log('🚀 Autism Assessment Tool initialized with backend integration');
    console.log(`📊 Form tracking: ${totalQuestions} questions + ${totalFields} demographics = ${totalFormFields} total fields`);
});
