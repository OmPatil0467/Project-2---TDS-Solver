from fastapi import FastAPI, File, Form, UploadFile, HTTPException
import shutil
import zipfile
import pandas as pd
import openai
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set OpenAI API key from environment variables
openai.api_key = os.getenv("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")

@app.post("/api/")
async def answer_question(
    question: str = Form(...), file: UploadFile = File(None)
):
    """
    API Endpoint to process a question and optional file upload.
    If a ZIP file is uploaded containing a CSV, extract the "answer" column.
    If only a text question is provided, use OpenAI's API to generate an answer.
    """
    if file:
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / file.filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if file.filename.endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
                extracted_files = list(temp_dir.glob("*.csv"))
                if extracted_files:
                    df = pd.read_csv(extracted_files[0])
                    if "answer" in df.columns:
                        answer = str(df["answer"].iloc[0])
                        return {"answer": answer}
        
        return {"answer": "File processed, but no 'answer' column found."}
    
    # Validate OpenAI API key
    if not openai.api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key is missing.")
    
    # Process text-based questions with OpenAI
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": question}]
        )
        return {"answer": response["choices"][0]["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Ensure FastAPI runs when executing the script directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
