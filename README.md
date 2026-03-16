# StudyForge AI Planner (Flask + Gemini)

A smart study planner website with:
- Personalized study plan generation
- AI doubt-solving chat assistant
- Quiz generator with answer key
- Document upload and AI summarization (TXT, MD, PDF, DOCX)
- AI flashcard generator
- 7-day revision rescue planner
- Download outputs (plan, quiz, summary, flashcards, chat)

## Tech Stack
- Python + Flask
- Gemini API via Google Generative AI SDK
- HTML/CSS/JavaScript frontend

## Quick Start
1. Create and activate a virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Configure environment variables:

   cp .env.example .env

   Then set your API key in .env

4. Run:

   python app.py

5. Open:

   http://127.0.0.1:5000

## Environment Variables
- GOOGLE_API_KEY: Your Gemini API key
- PORT: Flask port (default 5000)

Model selection is hardcoded in the backend to: gemini-3.1-flash-lite-preview

## API Endpoints
- POST /api/study-plan
- POST /api/chat
- POST /api/quiz
- POST /api/flashcards
- POST /api/revision-plan
- POST /api/upload-summary
- GET /health
