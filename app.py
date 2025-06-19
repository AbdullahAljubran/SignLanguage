from flask import Flask, render_template, Response, jsonify, request 
from sign_language_rag import get_llm_answer, clear_cache
from ultralytics import YOLO
import cv2
import time
import threading
import atexit

app = Flask(__name__)

# Load YOLO model
model = YOLO("weights/best.pt")

# Translation dictionary (English to Arabic)
translation_dict = { 
    'analysis': 'التحليل',
    'appointment': 'موعد',
    'complaint': 'شكوى',
    'dispensing': 'اصرف',
    'emergency': 'الطوارئ',
    'fasting': 'الصيام',
    'medicine': 'دواء',
    'result': 'نتيجة',
    'how': 'كيف',
    'condition': 'شرط',
    'when': 'متى',
    'do': 'هل',
    'submit': 'أقدم',
    'where': 'أين',
    'book': 'حجز',
}

def translate_to_arabic(english_text):
    """Translate English text to Arabic"""
    return translation_dict.get(english_text.lower().strip(), english_text)

# Global variables
cap = None
detected_sign = ""
sentence_words = []
sentence_words_arabic = []
last_detection_time = 0
detection_cooldown = 3
detection_lock = threading.Lock()
camera_active = False
detection_active = False
latest_frame = None

# RAG processing variables
last_rag_sentence = ""
current_rag_result = {"question": "", "answer": ""}
rag_processing = False

# Choose meaningfulness method
# Options: "similarity", "keywords", "count_only", "hybrid", "none"
MEANINGFULNESS_METHOD = "similarity"  # Change this to switch methods

def initialize_camera():
    """Initialize camera with error handling"""
    global cap, camera_active
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera - trying again...")
            cap = cv2.VideoCapture(1)
        
        if cap.isOpened():
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
    global latest_frame, detected_sign, last_detection_time, sentence_words, sentence_words_arabic, detection_active

    while True:
        if (detection_active and latest_frame is not None and 
            time.time() - last_detection_time >= detection_cooldown):
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

def process_rag_async():
    """Background thread to process RAG with configurable meaningfulness check"""
    global last_rag_sentence, current_rag_result, rag_processing, sentence_words_arabic
    
    while True:
        try:
            with detection_lock:
                current_sentence = ' '.join(sentence_words_arabic)
            
            # Only process if sentence changed and has minimum words
            min_words = 2 if MEANINGFULNESS_METHOD == "none" else 3
            
            if (current_sentence != last_rag_sentence and 
                len(sentence_words_arabic) >= min_words and 
                not rag_processing):
                
                rag_processing = True
                print(f"🔄 Processing RAG with method '{MEANINGFULNESS_METHOD}': {current_sentence}")
                
                # Process with chosen method
                rag_result = get_llm_answer("", sentence_words_arabic.copy(), 
                                          meaningfulness_method=MEANINGFULNESS_METHOD)
                
                if isinstance(rag_result, dict):
                    current_rag_result = rag_result
                else:
                    current_rag_result = {"question": "", "answer": rag_result}
                
                last_rag_sentence = current_sentence
                rag_processing = False
                print(f"✅ RAG processing complete")
                
        except Exception as e:
            print(f"❌ RAG processing error: {e}")
            rag_processing = False
        
        time.sleep(0.5)

def cleanup_camera():
    """Cleanup camera resources"""
    global cap, camera_active
    if cap is not None:
        cap.release()
        camera_active = False
        print("📹 Camera closed")

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

            latest_frame = frame.copy()

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
    global detected_sign, sentence_words_arabic, detection_active, current_rag_result, rag_processing
    
    with detection_lock:
        sentence = ' '.join(sentence_words_arabic)
        
        question = current_rag_result.get("question", "")
        answer = current_rag_result.get("answer", "")
        
        if rag_processing and len(sentence_words_arabic) >= 2:
            answer = ""

        return jsonify({
            'detected_sign': detected_sign,
            'sentence': sentence,
            'sentence_words_arabic': sentence_words_arabic,  
            'question': question,
            'answer': answer,
            'word_count': len(sentence_words_arabic),
            'camera_status': camera_active,
            'detection_status': detection_active,
            'processing': rag_processing,
            'meaningfulness_method': MEANINGFULNESS_METHOD,
            'timestamp': time.time()
        })

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """API endpoint to toggle detection on/off"""
    global detection_active, detected_sign
    with detection_lock:
        detection_active = not detection_active
        if not detection_active:
            detected_sign = ""
    
    status = "started" if detection_active else "stopped"
    print(f"🔄 Detection {status}")
    return jsonify({
        'status': status,
        'detection_active': detection_active
    })

@app.route('/change_method', methods=['POST'])
def change_method():
    """NEW: Change meaningfulness checking method"""
    global MEANINGFULNESS_METHOD
    data = request.json
    new_method = data.get("method", "similarity")
    
    valid_methods = ["similarity", "keywords", "count_only", "hybrid", "none"]
    if new_method in valid_methods:
        MEANINGFULNESS_METHOD = new_method
        clear_cache()  # Clear cache when changing methods
        print(f"🔄 Changed meaningfulness method to: {new_method}")
        return jsonify({
            'status': 'success',
            'method': MEANINGFULNESS_METHOD
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Invalid method'
        }), 400

@app.route('/clear_detection')
def clear_detection():
    """API endpoint to clear current detection"""
    global detected_sign
    try:
        with detection_lock:
            detected_sign = ""
        return jsonify({'status': 'cleared'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/clear_sentence')
def clear_sentence():
    """API endpoint to clear the entire sentence"""
    global sentence_words, sentence_words_arabic, detected_sign, last_rag_sentence, current_rag_result
    try:
        with detection_lock:
            sentence_words = []
            sentence_words_arabic = []
            detected_sign = ""
            last_rag_sentence = ""
            current_rag_result = {"question": "", "answer": ""}
        
        clear_cache()
        return jsonify({'status': 'sentence_cleared'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/remove_last_word')
def remove_last_word():
    """API endpoint to remove the last word from sentence"""
    global sentence_words, sentence_words_arabic, last_rag_sentence, current_rag_result
    try:
        with detection_lock:
            if sentence_words_arabic:
                removed_word_arabic = sentence_words_arabic.pop()
                removed_word_english = sentence_words.pop() if sentence_words else ""
                
                last_rag_sentence = ""
                current_rag_result = {"question": "", "answer": ""}
                
                return jsonify({
                    'status': 'word_removed', 
                    'removed_word_arabic': removed_word_arabic,
                    'removed_word_english': removed_word_english
                })
            else:
                return jsonify({'status': 'no_words_to_remove'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/restart_camera')
def restart_camera():
    """API endpoint to restart camera"""
    try:
        cleanup_camera()
        time.sleep(1)
        success = initialize_camera()
        return jsonify({
            'status': 'restarted' if success else 'failed', 
            'camera_active': camera_active
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    try:
        print("🔧 Initializing camera...")
        if not initialize_camera():
            print("❌ Failed to start camera - please check connection")
            exit(1)

        detection_thread = threading.Thread(target=detect_sign_loop, daemon=True)
        detection_thread.start()
        
        rag_thread = threading.Thread(target=process_rag_async, daemon=True)
        rag_thread.start()

        print("🚀 Starting Flask application...")
        print("📹 Go to: http://localhost:5000")
        print(f"🧠 Meaningfulness method: {MEANINGFULNESS_METHOD}")
        print("🔍 Detection starts OFF - use toggle button to enable")

        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

    except KeyboardInterrupt:
        print("\n⏹️ Application stopped")
    except Exception as e:
        print(f"❌ Application error: {e}")
    finally:
        cleanup_camera()