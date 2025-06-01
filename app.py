#app.py 
from flask import Flask, render_template, Response, jsonify, request 
from sign_language_rag import get_llm_answer, reformulate_to_question
from ultralytics import YOLO
import cv2
import time
import threading
import json
import atexit

app = Flask(__name__)

# Load YOLO model
model = YOLO("best.pt")

# Translation dictionary (English to Arabic)
translation_dict = {
    'analysis': 'تحليل',
    'appointment': 'موعد',
    'complaint': 'شكوى',
    'dispensing': 'صرف',
    'emergency': 'طوارئ',
    'fasting': 'صيام',
    'medicine': 'دواء',
    'result': 'نتيجة'
}

def translate_to_arabic(english_text):
    """Translate English text to Arabic"""
    return translation_dict.get(english_text.lower(), english_text)

# Global variables
cap = None
detected_sign = ""
sentence_words = []  # List to store detected words for sentence building
sentence_words_arabic = []  # List to store Arabic translations
last_detection_time = 0
detection_cooldown = 3  # 3 -> 5 seconds cooldown
detection_lock = threading.Lock()
camera_active = False
latest_frame = None  # Global frame shared between threads

def initialize_camera():
    """Initialize camera with error handling"""
    global cap, camera_active
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera - trying again...")
            cap = cv2.VideoCapture(1)  # Try different camera index
        
        if cap.isOpened():
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            camera_active = True
            print("✅ Camera started successfully")
            return True
        else:
            print("❌ Failed to start camera")
            return False
    except Exception as e:
        print(f"❌ Camera error: {e}")
        return False
    
def detect_sign_loop():
    global latest_frame, detected_sign, last_detection_time, sentence_words, sentence_words_arabic

    while True:
        if latest_frame is not None and time.time() - last_detection_time >= detection_cooldown:
            try:
                frame = cv2.resize(latest_frame, (640, 480))
                results = model(frame)

                if len(results[0].boxes) > 0:
                    with detection_lock:
                        class_id = int(results[0].boxes[0].cls[0])
                        detected_sign_english = model.names[class_id]
                        detected_sign_arabic = translate_to_arabic(detected_sign_english)

                        if not sentence_words_arabic or sentence_words_arabic[-1] != detected_sign_arabic:
                            sentence_words.append(detected_sign_english)
                            sentence_words_arabic.append(detected_sign_arabic)

                        detected_sign = detected_sign_arabic
                        last_detection_time = time.time()
                        print(f"✅ Detected: {detected_sign_english} → {detected_sign_arabic}")

            except Exception as e:
                print(f"❌ Detection error: {e}")

        time.sleep(0.1)

def cleanup_camera():
    """Cleanup camera resources"""
    global cap, camera_active
    if cap is not None:
        cap.release()
        camera_active = False
        print("📹 Camera closed")

# Register cleanup function
atexit.register(cleanup_camera)

def generate_frames():
    global latest_frame

    while camera_active:
        if cap is None:
            time.sleep(0.1)
            continue

        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            latest_frame = frame.copy()  # Update shared frame

            # Encode and stream
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"❌ Frame error: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    if not camera_active:
        return Response("Camera not available", mimetype='text/plain')
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detection')
def get_detection():
    global detected_sign, sentence_words_arabic
    with detection_lock:
        sentence = ' '.join(sentence_words_arabic)
        rag_result = get_llm_answer("", sentence_words_arabic) if sentence_words_arabic else ""

        if isinstance(rag_result, dict):
            question = rag_result["question"]
            answer = rag_result["answer"]
        else:
            question = ""
            answer = rag_result

        return jsonify({
            'detected_sign': detected_sign,
            'sentence': sentence,
            'question': question,
            'answer': answer,
            'word_count': len(sentence_words_arabic),
            'camera_status': camera_active,
            'timestamp': time.time()
        })

@app.route('/clear_detection')
def clear_detection():
    """API endpoint to clear current detection"""
    global detected_sign
    with detection_lock:
        detected_sign = ""
    return jsonify({'status': 'cleared'})

@app.route('/clear_sentence')
def clear_sentence():
    """API endpoint to clear the entire sentence"""
    global sentence_words, sentence_words_arabic, detected_sign
    with detection_lock:
        sentence_words = []
        sentence_words_arabic = []
        detected_sign = ""
    return jsonify({'status': 'sentence_cleared'})

@app.route('/ask_llm', methods=['POST'])
def ask_llm():
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer = get_llm_answer(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/remove_last_word')
def remove_last_word():
    """API endpoint to remove the last word from sentence"""
    global sentence_words, sentence_words_arabic
    with detection_lock:
        if sentence_words_arabic:
            removed_word_arabic = sentence_words_arabic.pop()
            removed_word_english = sentence_words.pop() if sentence_words else ""
            return jsonify({
                'status': 'word_removed', 
                'removed_word_arabic': removed_word_arabic,
                'removed_word_english': removed_word_english
            })
    return jsonify({'status': 'no_words_to_remove'})

@app.route('/restart_camera')
def restart_camera():
    """API endpoint to restart camera"""
    cleanup_camera()
    time.sleep(1)
    success = initialize_camera()
    return jsonify({'status': 'restarted' if success else 'failed', 'camera_active': camera_active})

if __name__ == '__main__':
    try:
        print("🔧 Initializing camera...")
        if not initialize_camera():
            print("❌ Failed to start camera - please check connection")
            exit(1)

        # ✅ Start YOLO detection in a background thread BEFORE app.run()
        detection_thread = threading.Thread(target=detect_sign_loop, daemon=True)
        detection_thread.start()

        print("🚀 Starting Flask application...")
        print("📹 Go to: http://localhost:5000")
        print("🛑 To stop the application press Ctrl+C")
        print("📱 Camera is running continuously")

        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

    except KeyboardInterrupt:
        print("\n⏹️ Application stopped")
    except Exception as e:
        print(f"❌ Application error: {e}")
    finally:
        cleanup_camera()