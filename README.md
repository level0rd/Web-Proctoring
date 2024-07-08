# Online proctoring system
Simple web application analyzes user actions for video proctoring systems. It automatically tracks the user’s gaze direction and hand position. The system detects the user’s departure and the presence of a second person. The presence of a voice is also detects.

## Build and run the application locally
```bash
streamlit run main.py
```

## Build and run the application in Docker
```bash
# Build Docker image
docker build . -t streamlit
# Run Docker container
docker run -p 8501:8501 streamlit
```

If the Docker container is running locally, the application will be available at http://localhost:8501.

## Architecture
<p align="center">
  <img width="4096" alt="Arch3_1" src="https://github.com/level0rd/Web-Proctoring/assets/45522296/918cddd3-3bb8-46d6-8c41-f9a45e68c53b.png">
</p>

## Examples
<p align="center">
  <img width="1000" alt="Arch2" src="https://github.com/level0rd/Web-Proctoring/assets/45522296/0bbca3c3-3540-429f-a66f-6e10a7035505.png">
</p>

### User leaving the frame
![Leaving](https://github.com/level0rd/Web-Proctoring/assets/45522296/5575fcf5-e058-4df0-a10d-389faa95affc)

### The second person in the frame
![Second_person](https://github.com/level0rd/Web-Proctoring/assets/45522296/2406d509-ddc6-490b-a988-04f0ef00b3d6)

### The direction of view
![Direction_of_view](https://github.com/level0rd/Web-Proctoring/assets/45522296/3f210f53-cbbf-4fad-af2e-fb7e095f41d0)

### Recording of the detected second person in the frame onto the video file
![Double_person_v2](https://github.com/level0rd/Web-Proctoring/assets/45522296/06bf1ffa-c224-4b58-8fec-5b73cc671e12)
