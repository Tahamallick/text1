from fastapi import APIRouter, Request, HTTPException
from datetime import datetime
import json
from fastapi.responses import RedirectResponse

router = APIRouter()

@router.get("/latest-transcript{path:path}")
async def get_latest_transcript(request: Request, path: str = ""):
    """Get the most recent transcript and its status"""
    # If path is not empty (contains spaces or other characters), redirect to clean URL
    if path.strip():
        return RedirectResponse(url="/latest-transcript", status_code=301)
        
    transcription_service = request.state.transcription_service
    try:
        print("\nChecking for latest transcript...")
        response = transcription_service.s3_client.list_objects_v2(
            Bucket=transcription_service.bucket_name,
            Prefix="transcripts/",
            MaxKeys=10
        )
        
        if 'Contents' not in response:
            return {
                "status": "success",
                "message": "No transcripts found",
                "latest_transcript": None,
                "is_processing": False,
                "timestamp": datetime.now().isoformat()
            }
            
        # Sort by last modified, newest first
        files = sorted(
            response['Contents'],
            key=lambda x: x['LastModified'],
            reverse=True
        )
        
        # Get the latest transcript file
        latest_file = None
        for file in files:
            if file['Key'].endswith('.txt') or file['Key'].endswith('.json'):
                latest_file = file
                break
                
        if not latest_file:
            return {
                "status": "success",
                "message": "No transcript files found",
                "latest_transcript": None,
                "is_processing": False,
                "timestamp": datetime.now().isoformat()
            }
            
        # Check if there are any audio files being processed
        audio_response = transcription_service.s3_client.list_objects_v2(
            Bucket=transcription_service.bucket_name,
            Prefix="meetings/"
        )
        
        is_processing = False
        if 'Contents' in audio_response:
            for obj in audio_response['Contents']:
                if obj['Key'].endswith('.wav'):
                    is_processing = True
                    break
        
        # Get the transcript content
        try:
            content_response = transcription_service.s3_client.get_object(
                Bucket=transcription_service.bucket_name,
                Key=latest_file['Key']
            )
            
            content = content_response['Body'].read().decode('utf-8')
            
            # If it's a JSON file, try to extract the transcript text
            if latest_file['Key'].endswith('.json'):
                try:
                    json_content = json.loads(content)
                    if 'results' in json_content:
                        content = json_content['results'].get('transcripts', [{}])[0].get('transcript', content)
                except:
                    pass
            
            return {
                "status": "success",
                "message": "Latest transcript retrieved",
                "latest_transcript": {
                    "key": latest_file['Key'],
                    "content": content,
                    "last_modified": latest_file['LastModified'].isoformat()
                },
                "is_processing": is_processing,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error reading latest transcript: {e}")
            return {
                "status": "error",
                "message": "Error reading latest transcript",
                "error": str(e),
                "is_processing": is_processing,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        print(f"Error in get_latest_transcript: {e}")
        return {
            "status": "error",
            "message": "Error checking latest transcript",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        } 