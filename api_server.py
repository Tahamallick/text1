from fastapi import FastAPI, HTTPException, BackgroundTasks, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from transcription_service import TranscriptionService
import os
from dotenv import load_dotenv
import asyncio
from datetime import datetime
import threading
import time
from contextlib import asynccontextmanager
from pyngrok import ngrok, conf
from latest_endpoint import router as latest_router
import requests
import json
import logging
import boto3
from botocore.exceptions import ClientError
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Load environment variables
load_dotenv()

# Configure ngrok
conf.get_default().config_path = os.path.join(os.path.expanduser("~"), ".ngrok2", "ngrok.yml")
conf.get_default().region = "us"  # or your preferred region

# Set ngrok authtoken from environment variable
ngrok_token = os.getenv('NGROK_AUTHTOKEN')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_ngrok():
    """Setup ngrok with proper error handling"""
    try:
        # Check if there are any existing tunnels and disconnect them
        try:
            tunnels = ngrok.get_tunnels()
            for tunnel in tunnels:
                print(f"Disconnecting existing tunnel: {tunnel.public_url}")
                ngrok.disconnect(tunnel.public_url)
        except Exception as e:
            print(f"Error cleaning up existing tunnels: {e}")

        # Configure basic auth if credentials are provided
        auth_token = os.getenv('NGROK_BASIC_AUTH')  # format: "username:password"
        options = {
            "bind_tls": True,  # Enable HTTPS
        }
        if auth_token:
            options["auth"] = auth_token

        # Create new tunnel with options
        tunnel = ngrok.connect(8000, "http", options=options)
        print("\n=== IMPORTANT: NGROK ACCESS INSTRUCTIONS ===")
        print(f"Public URL: {tunnel.public_url}")
        print("\nFor first-time access:")
        print("1. Open the ngrok URL in a browser")
        print("2. Click 'Visit Site' on the warning page")
        print("3. After that, API calls will work normally")
        print("\nShare these instructions with anyone accessing the API")
        print("==========================================\n")
        return tunnel
    except Exception as e:
        print("\n=== NGROK SETUP ERROR ===")
        print(f"Error setting up ngrok: {e}")
        print("\nTry these solutions:")
        print("1. Use local network instead:")
        print("   - Run 'ipconfig' to find your IP")
        print("   - Share http://YOUR_IP:8000")
        print("2. Or upgrade to ngrok paid plan")
        print("========================\n")
        return None

# Global variables
last_check_time = None
is_checking_active = False
is_arduino_checking_active = False  # New flag to control Arduino checking
checking_thread = None
arduino_check_thread = None
ngrok_tunnel = None
ARDUINO_API_URL = "http://192.168.0.200/status"
shutdown_event = threading.Event()

def check_arduino_status():
    """Check Arduino status periodically"""
    global is_checking_active, checking_thread, is_arduino_checking_active
    last_status = None
    
    print("\n[Arduino Check] Starting Arduino status monitoring...")
    
    while not shutdown_event.is_set():
        try:
            print(f"\n[Arduino Status Check] Attempting to connect to {ARDUINO_API_URL}")
            response = requests.get(ARDUINO_API_URL, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', '')
                meeting_count = data.get('meetingCount', 0)
                
                print(f"Status: {status}")
                print(f"Meeting Count: {meeting_count}")
                print(f"Audio Checking Active: {is_checking_active}")
                
                # Handle status changes
                if status == "recording":
                    if not is_checking_active:
                        print("\n=== Recording Started ===")
                        print("Starting audio file monitoring...")
                        is_checking_active = True
                        if checking_thread and checking_thread.is_alive():
                            checking_thread.join()
                        checking_thread = threading.Thread(target=check_for_new_audio_periodically)
                        checking_thread.daemon = True
                        checking_thread.start()
                elif status == "stopped":
                    if is_checking_active:
                        print("\n=== Recording Stopped ===")
                        print("Pausing audio file monitoring...")
                        is_checking_active = False
                        if checking_thread and checking_thread.is_alive():
                            checking_thread.join()
                        print("Waiting for next recording signal...")
                
                last_status = status
            else:
                print(f"Error: Arduino returned status code {response.status_code}")
                print("Continuing with last known status...")
                
        except requests.exceptions.Timeout:
            print("Arduino connection timed out. Continuing with last known status...")
        except requests.exceptions.ConnectionError:
            print("Could not connect to Arduino. Please check:")
            print("1. The Arduino is powered on")
            print("2. The IP address is correct (currently set to 192.168.1.162)")
            print("3. The Arduino is on the same network as this computer")
            print("4. No firewall is blocking the connection")
            print("Continuing with last known status...")
        except Exception as e:
            print(f"Error checking Arduino status: {e}")
            print("Continuing with last known status...")
        
        time.sleep(5)  # Check every 5 seconds
    
    print("\n[Arduino Check] Arduino status monitoring stopped")

def run_transcription(audio_key):
    """Run transcription in a separate thread"""
    try:
        print(f"\n[Transcription Start]")
        print(f"Audio Key: {audio_key}")
        print("Starting transcription process...")
        
        # Check if file exists before attempting transcription
        try:
            transcription_service.s3_client.head_object(
                Bucket=transcription_service.bucket_name,
                Key=audio_key
            )
        except Exception as e:
            print(f"Audio file {audio_key} not found in S3: {e}")
            return
        
        print("Audio file found in S3, starting transcription...")
        result = transcription_service.transcribe_audio(audio_key)
        
        if result:
            print(f"Transcription completed successfully for: {audio_key}")
            # Delete the meeting file after successful transcription
            try:
                print(f"Deleting meeting file: {audio_key}")
                transcription_service.s3_client.delete_object(
                    Bucket=transcription_service.bucket_name,
                    Key=audio_key
                )
                print(f"Meeting file deleted successfully: {audio_key}")
                
                # Check meeting status after successful transcription
                try:
                    response = requests.get(ARDUINO_API_URL, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        meeting_count = data.get('meetingCount', 0)
                        status = data.get('status', '')
                        
                        print(f"\n[Post-Transcription Status]")
                        print(f"Status: {status}")
                        print(f"Meeting Count: {meeting_count}")
                        print(f"Transcription Result: {result}")
                        
                        if meeting_count == 0:
                            print("No meetings active, stopping audio checking")
                            global is_checking_active
                            is_checking_active = False
                except Exception as e:
                    print(f"Error checking meeting status after transcription: {e}")
            except Exception as e:
                print(f"Error deleting meeting file: {e}")
        else:
            print(f"Transcription failed for: {audio_key}")
            print("Please check the following:")
            print("1. The audio file exists in S3")
            print("2. The audio file format is supported")
            print("3. The AWS credentials are correct")
            print("4. The transcription service is working")
    except Exception as e:
        print(f"Error in transcription thread: {e}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())

def check_for_new_audio_periodically():
    """Background task to check for new audio files every 10 seconds"""
    global last_check_time
    print("\n[Audio Check] Starting audio file monitoring")
    last_processed_files = set()
    failed_files = {}  # Track failed files and their retry count
    
    while is_checking_active and not shutdown_event.is_set():
        try:
            print(f"\n[{datetime.now()}] Checking for new audio files...")
            try:
                # List all files in the meetings directory
                response = transcription_service.s3_client.list_objects_v2(
                    Bucket=transcription_service.bucket_name,
                    Prefix="meetings/"
                )
                
                if 'Contents' in response:
                    current_files = set()
                    for obj in response['Contents']:
                        audio_key = obj['Key']
                        if audio_key.endswith('.wav'):  # Process only WAV files
                            current_files.add(audio_key)
                            
                            # Check if file is new or previously failed
                            if audio_key not in last_processed_files or audio_key in failed_files:
                                retry_count = failed_files.get(audio_key, 0)
                                if retry_count < 3:  # Allow up to 3 retries
                                    print(f"Processing audio file: {audio_key} (Attempt {retry_count + 1})")
                                    # Start transcription in a separate thread
                                    thread = threading.Thread(
                                        target=run_transcription_with_retry,
                                        args=(audio_key, failed_files)
                                    )
                                    thread.daemon = True
                                    thread.start()
                                    print(f"Started transcription for: {audio_key}")
                                else:
                                    print(f"Skipping {audio_key} after {retry_count} failed attempts")
                    
                    # Update the set of processed files
                    last_processed_files = current_files
                else:
                    print("No files found in meetings directory")
                    last_processed_files = set()
                    
            except Exception as e:
                print(f"Error checking for new audio: {e}")
                import traceback
                print(traceback.format_exc())
            
            last_check_time = datetime.now()
            
        except Exception as e:
            print(f"Error in background check: {e}")
            import traceback
            print("Full error traceback:")
            print(traceback.format_exc())
            
        if not is_checking_active:
            print("\n[Audio Check] Stopping audio file monitoring - Recording stopped")
            break
            
        time.sleep(10)  # Check every 10 seconds
    
    print("\n[Audio Check] Audio file monitoring stopped")

def run_transcription_with_retry(audio_key, failed_files):
    """Run transcription with retry tracking"""
    try:
        print(f"\n[Transcription Start]")
        print(f"Audio Key: {audio_key}")
        print("Starting transcription process...")
        
        # Verify file exists and is accessible
        try:
            obj = transcription_service.s3_client.head_object(
                Bucket=transcription_service.bucket_name,
                Key=audio_key
            )
            file_size = obj['ContentLength']
            print(f"File size: {file_size} bytes")
            if file_size == 0:
                print(f"Warning: {audio_key} is empty")
                failed_files[audio_key] = failed_files.get(audio_key, 0) + 1
                return
        except Exception as e:
            print(f"Error accessing file {audio_key}: {e}")
            failed_files[audio_key] = failed_files.get(audio_key, 0) + 1
            return
        
        # Start transcription
        result = transcription_service.transcribe_audio(audio_key)
        
        if result:
            print(f"Transcription completed successfully for: {audio_key}")
            # Remove from failed files if it was there
            failed_files.pop(audio_key, None)
            
            # Delete the meeting file after successful transcription
            try:
                print(f"Deleting meeting file: {audio_key}")
                transcription_service.s3_client.delete_object(
                    Bucket=transcription_service.bucket_name,
                    Key=audio_key
                )
                print(f"Meeting file deleted successfully: {audio_key}")
                
                # Check Arduino status after successful transcription
                try:
                    response = requests.get(ARDUINO_API_URL, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        status = data.get('status', '')
                        
                        print(f"\n[Post-Transcription Status]")
                        print(f"Arduino Status: {status}")
                        print(f"Transcription Result: {result}")
                        
                except Exception as e:
                    print(f"Error checking Arduino status after transcription: {e}")
            except Exception as e:
                print(f"Error deleting meeting file: {e}")
        else:
            print(f"Transcription failed for: {audio_key}")
            failed_files[audio_key] = failed_files.get(audio_key, 0) + 1
            print(f"Failed attempts for {audio_key}: {failed_files[audio_key]}")
            print("Please check the following:")
            print("1. The audio file exists in S3")
            print("2. The audio file format is supported")
            print("3. The AWS credentials are correct")
            print("4. The transcription service is working")
    except Exception as e:
        print(f"Error in transcription thread: {e}")
        failed_files[audio_key] = failed_files.get(audio_key, 0) + 1
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global checking_thread, arduino_check_thread, ngrok_tunnel, shutdown_event, is_arduino_checking_active
    
    # Reset shutdown event
    shutdown_event.clear()
    
    # Start ngrok tunnel
    if ngrok_token:
        ngrok_tunnel = setup_ngrok()
    else:
        print("\n=== RUNNING WITHOUT NGROK ===")
        print("No ngrok authtoken provided. Server will be accessible:")
        print("1. Locally at: http://localhost:8000")
        print("2. On your network at: http://YOUR_LOCAL_IP:8000")
        print("To get your local IP, run:")
        print("- Windows: ipconfig")
        print("- Linux/Mac: ifconfig or ip addr")
        print("===========================\n")
    
    # Start Arduino status checking only when needed
    is_arduino_checking_active = True
    arduino_check_thread = threading.Thread(target=check_arduino_status)
    arduino_check_thread.daemon = True
    arduino_check_thread.start()
    print("Arduino status checking started")
    
    yield
    
    # Shutdown
    print("\nShutting down server...")
    shutdown_event.set()
    is_arduino_checking_active = False
    
    if ngrok_tunnel:
        try:
            ngrok.disconnect(ngrok_tunnel.public_url)
            print("ngrok tunnel closed")
        except Exception as e:
            print(f"Error closing ngrok tunnel: {e}")
    
    # Stop all background threads
    is_checking_active = False
    
    if checking_thread and checking_thread.is_alive():
        checking_thread.join(timeout=5)
        if checking_thread.is_alive():
            print("Warning: Audio checking thread did not stop gracefully")
    
    if arduino_check_thread and arduino_check_thread.is_alive():
        arduino_check_thread.join(timeout=5)
        if arduino_check_thread.is_alive():
            print("Warning: Arduino status checking thread did not stop gracefully")
    
    print("Server shutdown complete")

app = FastAPI(title="Transcription Service API", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Add route to serve test.html
@app.get("/test.html")
async def serve_test_html():
    return FileResponse("test.html")

# Add CORS middleware with specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=False,  # Set to False since we're using allow_origins=["*"]
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Max-Age"] = "3600"
    return response

# Initialize transcription service
transcription_service = TranscriptionService()

# Include the latest-transcript endpoint
app.include_router(latest_router)

# Pass transcription service to the router
@app.middleware("http")
async def add_transcription_service(request, call_next):
    request.state.transcription_service = transcription_service
    response = await call_next(request)
    return response

class TranscriptionResponse(BaseModel):
    status: str
    message: str
    transcript: Optional[str] = None
    error: Optional[str] = None

class AudioFileResponse(BaseModel):
    status: str
    message: str
    file_key: Optional[str] = None
    error: Optional[str] = None

class MeetingStatus(BaseModel):
    status: str
    message: str
    is_active: bool
    error: Optional[str] = None

class ArduinoSignal(BaseModel):
    meeting_id: str
    start_time: str
    status: str

@app.get("/")
async def root():
    return {"message": "Transcription Service API is running"}

@app.post("/arduino/meeting-start", response_model=MeetingStatus)
async def arduino_meeting_start(signal: ArduinoSignal):
    """Receive meeting start signal from Arduino"""
    global is_checking_active, checking_thread
    
    if is_checking_active:
        return {
            "status": "success",
            "message": "Meeting is already active",
            "is_active": True
        }
    
    try:
        print(f"Received meeting start signal from Arduino: {signal.meeting_id}")
        print(f"Meeting start time: {signal.start_time}")
        
        is_checking_active = True
        checking_thread = threading.Thread(target=check_for_new_audio_periodically)
        checking_thread.daemon = True
        checking_thread.start()
        print("Meeting started by Arduino - Audio checking activated")
        
        return {
            "status": "success",
            "message": "Meeting started successfully",
            "is_active": True
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "Failed to start meeting",
            "is_active": False,
            "error": str(e)
        }

@app.post("/arduino/meeting-stop", response_model=MeetingStatus)
async def arduino_meeting_stop(signal: ArduinoSignal):
    """Receive meeting stop signal from Arduino"""
    global is_checking_active, checking_thread
    
    if not is_checking_active:
        return {
            "status": "success",
            "message": "Meeting is already inactive",
            "is_active": False
        }
    
    try:
        print(f"Received meeting stop signal from Arduino: {signal.meeting_id}")
        print(f"Meeting stop time: {signal.start_time}")
        
        is_checking_active = False
        if checking_thread and checking_thread.is_alive():
            checking_thread.join()
        print("Meeting stopped by Arduino - Audio checking deactivated")
        
        return {
            "status": "success",
            "message": "Meeting stopped successfully",
            "is_active": False
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "Failed to stop meeting",
            "is_active": True,
            "error": str(e)
        }

@app.get("/meeting-status", response_model=MeetingStatus)
async def get_meeting_status():
    """Get the current status of the meeting"""
    return {
        "status": "success",
        "message": "Meeting status retrieved",
        "is_active": is_checking_active
    }

@app.get("/check-new-audio", response_model=AudioFileResponse)
async def check_new_audio():
    """Check for new audio files in the S3 bucket"""
    if not is_checking_active:
        return {
            "status": "error",
            "message": "Meeting is not active. Please start the meeting first.",
            "error": "Meeting inactive"
        }
    
    try:
        audio_key = transcription_service.check_new_audio()
        if audio_key:
            return {
                "status": "success",
                "message": "New audio file found",
                "file_key": audio_key
            }
        return {
            "status": "success",
            "message": "No new audio files found"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "Error checking for new audio",
            "error": str(e)
        }

@app.post("/transcribe/{audio_key}", response_model=TranscriptionResponse)
async def transcribe_audio(audio_key: str, background_tasks: BackgroundTasks):
    """Transcribe a specific audio file"""
    try:
        # Start transcription in background
        background_tasks.add_task(transcription_service.transcribe_audio, audio_key)
        
        return {
            "status": "success",
            "message": "Transcription started",
            "transcript": None
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "Error starting transcription",
            "error": str(e)
        }

@app.get("/list-transcripts")
async def list_transcripts():
    """List all available transcripts"""
    try:
        print("\nListing transcripts from S3...")
        
        # List all transcript files
        response = transcription_service.s3_client.list_objects_v2(
            Bucket=transcription_service.bucket_name,
            Prefix="transcripts/"
        )
        
        transcripts = []
        if 'Contents' in response:
            print(f"Found {len(response['Contents'])} objects in S3")
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.txt'):  # Only return .txt files
                    try:
                        # Get the transcript content
                        content_response = transcription_service.s3_client.get_object(
                            Bucket=transcription_service.bucket_name,
                            Key=key
                        )
                        content = content_response['Body'].read().decode('utf-8')
                        
                        # Format response to match mobile app expectations
                        transcripts.append({
                            "key": key,
                            "exact_url": f"/get-transcript/{key.replace('transcripts/', '').replace('.txt', '')}",
                            "last_modified": obj['LastModified'].isoformat(),
                            "content": content,
                            "size": obj['Size']
                        })
                    except Exception as e:
                        print(f"Error reading transcript {key}: {e}")
                        continue
        
        # Sort transcripts by last_modified, newest first
        transcripts.sort(key=lambda x: x['last_modified'], reverse=True)
        
        print(f"Returning {len(transcripts)} transcript(s)")
        
        # Return a simple JSONResponse with proper headers
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Found {len(transcripts)} transcript(s)",
                "transcripts": transcripts
            }
        )
            
    except Exception as e:
        print(f"Error listing transcripts: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "transcripts": []
            }
        )

@app.get("/get-transcript/{transcript_id}")
async def get_transcript(transcript_id: str):
    """Get a specific transcript"""
    try:
        print(f"\nFetching transcript: {transcript_id}")
        
        # Build the key
        transcript_key = f"transcripts/{transcript_id}"
        if not transcript_key.endswith('.txt'):
            transcript_key += '.txt'
        
        try:
            response = transcription_service.s3_client.get_object(
                Bucket=transcription_service.bucket_name,
                Key=transcript_key
            )
            content = response['Body'].read().decode('utf-8')
            
            # Format response to match mobile app expectations
            return {
                "status": "success",
                "message": "Transcript retrieved successfully",
                "transcript": content,
                "key": transcript_id,
                "last_modified": response['LastModified'].isoformat()
            }
            
        except transcription_service.s3_client.exceptions.NoSuchKey:
            print(f"Transcript not found: {transcript_key}")
            return {
                "status": "error",
                "message": f"Transcript not found: {transcript_key}",
                "transcript": None
            }
            
    except Exception as e:
        print(f"Error retrieving transcript: {e}")
        return {
            "status": "error",
            "message": str(e),
            "transcript": None
        }

# Add a simple test endpoint for mobile clients
@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify API access"""
    return {
        "status": "ok",
        "message": "API is working"
    }

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("API_PORT", 8000))
    print(f"\nStarting server on port {port}")
    print("Make sure to run ngrok in a separate terminal with: ngrok http 8000")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port) 
    