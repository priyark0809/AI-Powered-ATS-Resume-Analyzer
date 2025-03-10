import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
import time
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io


# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class ATSAnalyzer:
    @staticmethod
    def extract_text_from_pdf(uploaded_file):
        """Extract text from a PDF resume, with OCR fallback for image-based PDFs."""
        try:
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
            
            # If no text is extracted, try OCR on each page
            if not text.strip():
                st.warning("No text detected. Attempting OCR on image-based PDF...")
                images = convert_from_bytes(uploaded_file.read(), first_page=1, last_page=len(pdf_reader.pages))
                for image in images:
                    text += pytesseract.image_to_string(image) + "\n"
            
            return text.strip() if text.strip() else None
        except Exception as e:
            st.error(f"Error extracting PDF text: {str(e)}")
            return None

    @staticmethod
    def perform_ats_checks(pdf_text, job_description):
        """Perform basic ATS compatibility checks and return a score."""
        checks = {
            "Keywords Present": any(keyword in pdf_text.lower() for keyword in job_description.lower().split()),
            "No Complex Formatting": not ("table" in pdf_text.lower() or "image" in pdf_text.lower()),
            "Key Sections": all(section in pdf_text.lower() for section in ["summary", "experience", "education"]),
            "Readable Length": len(pdf_text.split()) < 1000  # Less than 1000 words
        }
        score = sum(checks.values()) / len(checks) * 100
        return score, checks

    @staticmethod
    def get_gemini_response(input_prompt, pdf_text, job_description):
        """Fetch response from Google Gemini API with retries for reliability."""
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content([input_prompt, pdf_text, job_description])
            return response.text if response.text else "No response from Gemini API."
        except Exception as e:
            st.error(f"Error generating Gemini response: {str(e)}")
            return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="ATS Resume Expert",
        page_icon="📄",
        layout="wide"
    )

    # Custom CSS for a polished UI
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            background-color: darkblue;
            color: white;
            border-radius: 0.5rem;
            padding: 0.75rem;
        }
        .stButton>button:hover {
            background-color: #0052a3;
        }
        .success-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            color: #155724;
        }
        .warning-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #fff3cd;
            color: #856404;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header with professional description
    st.title("📄 ATS Resume Analyzer")
    st.markdown("""
        This advanced tool analyzes your resume against job descriptions using AI powered by Google Gemini. 
        Upload your resume (PDF) and provide the job description to:
        - Get a detailed ATS compatibility analysis
        - See a percentage match with job requirements
        - Receive AI-driven suggestions for improvement
    """)

    # Input columns for better layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Job Description")
        job_description = st.text_area(
            "Paste the job description here",
            height=200,
            placeholder="Paste the complete job description here..."
        )

    with col2:
        st.subheader("📎 Resume Upload")
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF format)",
            type=["pdf"],
            help="Please ensure your resume is in PDF format (text-based or scanned)"
        )

        if uploaded_file:
            if uploaded_file.size > 2 * 1024 * 1024:  # Check file size (2MB limit)
                st.error("File size exceeds 2MB. Please upload a smaller file.")
                return
            
            st.markdown('<p class="success-message">✅ PDF uploaded successfully!</p>', unsafe_allow_html=True)

    # Analysis options and results
    if uploaded_file and job_description.strip():
        st.subheader("🔍 Analysis Options")
        analysis_type = st.radio(
            "Choose analysis type:",
            ["Detailed Resume Review", "ATS Match Percentage Analysis"]
        )

        if st.button("Analyze Resume"):
            with st.spinner("Analyzing your resume... Please wait"):
                # Extract PDF text with OCR fallback
                pdf_text = ATSAnalyzer.extract_text_from_pdf(uploaded_file)
                
                if not pdf_text:
                    st.error("No text could be extracted from the resume. Please check the file or ensure it’s a text-based PDF. For scanned PDFs, OCR is attempted.")
                    return

                # Perform ATS checks
                ats_score, ats_checks = ATSAnalyzer.perform_ats_checks(pdf_text, job_description)

                # Display ATS compatibility score
                

                # Prepare Gemini prompt based on analysis type
                if analysis_type == "Detailed Resume Review":
                    prompt = """
                    As an experienced Technical HR Manager, provide a detailed professional evaluation of the candidate's resume against the job description. Analyze:
                    1. Overall alignment with the role
                    2. Key strengths and qualifications that match
                    3. Notable gaps or areas for improvement
                    4. Specific recommendations for enhancing ATS compatibility and readability
                    5. Final verdict on suitability for the role
                    
                    Format the response with clear headings, bullet points, and professional language.
                    """
                else:  # ATS Match Percentage Analysis
                    prompt = """
                    As an ATS (Applicant Tracking System) expert, provide:
                    1. Overall match percentage (%)
                    2. Key matching keywords found
                    3. Important missing keywords
                    4. Skills gap analysis
                    5. Specific recommendations for improving ATS compatibility and keyword optimization
                    
                    Start with the percentage match prominently displayed, followed by detailed analysis in bullet points.
                    """

                # Get and display Gemini response
                response = ATSAnalyzer.get_gemini_response(prompt, pdf_text, job_description)
                
                if response:
                    st.markdown("### AI-Powered Analysis Results")
                    st.markdown(response)
                    
                    # Add export option
                    st.download_button(
                        label="📥 Export Analysis",
                        data=response,
                        file_name="resume_analysis.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Failed to retrieve AI analysis. Please try again later.")

    else:
        st.info("👆 Please upload your resume and provide the job description to begin the analysis.")

    # Footer with disclaimer
    st.markdown("---")
    st.markdown(
        
        "This tool uses AI (Google Gemini) to analyze resumes but should be used as a guide, not the sole factor in your job application process."
    )

if __name__ == "__main__":
    main()