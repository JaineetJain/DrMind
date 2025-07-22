# Dr. Mind - AI-Powered Mood Tracker

A Flask-based web application for mood tracking with AI-powered insights.

## Features

- User registration and login with secure password validation
- Mood tracking with 20 different emotional states
- AI-powered insights using Hugging Face API
- Sentiment analysis and mood visualization
- Responsive design for desktop and mobile

## Deployment on Render

### Prerequisites

1. A Render account (free tier available)
2. A GitHub repository with this code

### Steps to Deploy

1. **Create a PostgreSQL Database on Render:**
   - Go to your Render dashboard
   - Click "New" → "PostgreSQL"
   - Choose a name for your database (e.g., "drmind-db")
   - Select the free tier
   - Click "Create Database"
   - Copy the "External Database URL" from the database info page

2. **Deploy the Web Service:**
   - Go to your Render dashboard
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Configure the service:
     - **Name:** dr-mind-app (or your preferred name)
     - **Environment:** Python 3
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `gunicorn app:app`
   - Add Environment Variable:
     - **Key:** `DATABASE_URL`
     - **Value:** [Paste the External Database URL from step 1]
   - Click "Create Web Service"

3. **Wait for Deployment:**
   - Render will automatically build and deploy your application
   - The process usually takes 2-5 minutes
   - You'll get a live URL once deployment is complete

### Environment Variables

The application requires one environment variable:

- `DATABASE_URL`: PostgreSQL connection string (automatically provided by Render)

### Local Development

To run locally:

```bash
pip install -r requirements.txt
python app.py
```

For local development, the app will use SQLite if no `DATABASE_URL` is provided.

## API Keys

The application includes hardcoded API keys for:
- Google AI Studio
- Hugging Face
- Cohere

These are configured for demonstration purposes. In production, consider using environment variables for API keys.

## Troubleshooting

If you encounter a 500 Internal Server Error:

1. Check the Render logs for detailed error messages
2. Ensure the `DATABASE_URL` environment variable is set correctly
3. Verify that the PostgreSQL database is running and accessible
4. Check that all dependencies are installed correctly

## Support

For issues or questions, please check the Render logs first, as they provide detailed error information.

