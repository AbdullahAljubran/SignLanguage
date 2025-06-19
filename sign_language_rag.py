#sign_language_rag.py - Pattern-Free Solution (OpenAI Removed)
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import time
import hashlib
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Predefined Q&A knowledge base
qa_pairs = []
with open("qa.txt", "r", encoding="utf-8-sig") as f:
    for line in f:
        if "|" in line:
            question, answer = line.strip().split("|", 1)
            question = question.strip().replace('\u200f', '').replace('\ufeff', '')
            answer = answer.strip()
            qa_pairs.append((question, answer))

questions = [q for q, _ in qa_pairs]
answers = [a for _, a in qa_pairs]

# Load multilingual sentence embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Create FAISS index
q_embeddings = model.encode(questions)
index = faiss.IndexFlatL2(q_embeddings.shape[1])
index.add(np.array(q_embeddings).astype(np.float32))

# CACHE SYSTEM
answer_cache = {}
last_processed_sentence = ""
last_result = {}
processing_lock = False
reformulation_cache = {}

def get_sentence_hash(sentence_words):
    """Create a unique hash for the sentence to use as cache key"""
    sentence = ' '.join(sentence_words)
    return hashlib.md5(sentence.encode('utf-8')).hexdigest()

# SOLUTION 1: Similarity-Based Approach (Recommended)
def is_meaningful_by_similarity(sentence_words, similarity_threshold=15.0):
    """
    Check if sentence is meaningful by measuring similarity to known questions
    No patterns needed - uses AI similarity scoring
    """
    if len(sentence_words) < 2:
        return False
    
    sentence = ' '.join(sentence_words)
    
    # Get similarity to all known questions
    input_embedding = model.encode([sentence])
    D, I = index.search(np.array(input_embedding).astype(np.float32), k=1)
    
    best_distance = D[0][0]
    
    # If similarity is reasonable, consider it meaningful
    # Lower distance = higher similarity
    print(f"🔍 Similarity check: '{sentence}' -> distance: {best_distance}")
    
    return best_distance < similarity_threshold

# SOLUTION 2: Length + Keywords Approach (Simple but effective)
def is_meaningful_by_length_keywords(sentence_words, min_words=3):
    """
    Simple approach: Check length + YOUR ACTUAL detectable words only
    More flexible than rigid patterns
    """
    if len(sentence_words) < min_words:
        return False
    
    sentence = ' '.join(sentence_words)
    
    # ONLY words from your translation_dict - medical/topic words
    medical_keywords = [
        'التحليل',   # analysis
        'موعد',      # appointment  
        'شكوى',      # complaint
        'اصرف',      # dispensing
        'الطوارئ',   # emergency
        'الصيام',    # fasting
        'دواء',      # medicine
        'نتيجة',     # result
        'شرط'        # condition
    ]
    
    # ONLY words from your translation_dict - question/action words
    action_keywords = [
        'كيف',       # how
        'متى',       # when
        'هل',        # do
        'أقدم',      # submit
        'أين',       # where
        'حجز'        # book
    ]
    
    # Check if sentence contains both medical and action terms
    has_medical = any(keyword in sentence for keyword in medical_keywords)
    has_action = any(keyword in sentence for keyword in action_keywords)
    
    return has_medical and has_action

# SOLUTION 3: Minimum Word Count Only (Simplest)
def is_meaningful_by_count_only(sentence_words, min_words=4):
    """
    Simplest approach: Just check minimum word count
    Assumes users will naturally form meaningful sentences
    """
    return len(sentence_words) >= min_words

# SOLUTION 4: Hybrid Approach (Combines multiple methods)
def is_meaningful_hybrid(sentence_words):
    """
    Combines similarity + keywords for best results
    """
    # Quick length check
    if len(sentence_words) < 3:
        return False
    
    # Try similarity first (most accurate)
    if is_meaningful_by_similarity(sentence_words, similarity_threshold=12.0):
        return True
    
    # Fallback to keywords (for edge cases)
    if len(sentence_words) >= 4 and is_meaningful_by_length_keywords(sentence_words, min_words=3):
        return True
    
    return False

# Fast question reformulation - pattern-free
def reformulate_to_question(sentence):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                        "أنت مساعد افتراضي يعمل داخل مستشفى، مهمتك إعادة صياغة الجملة التالية لتصبح "
                        "سؤالًا واضحًا باللغة العربية يتعلق بالخدمات أو الاستفسارات الطبية، "
                        "مع الالتزام بالكلمات الأصلية إن أمكن، وعدم إضافة معلومات جديدة أو خارجية."
                )
            },
            {"role": "user", "content": sentence}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# MAIN function with configurable meaningfulness check
def get_llm_answer(_, sentence_words_arabic, threshold=10.0, meaningfulness_method="similarity"):
    global last_processed_sentence, last_result, processing_lock
    
    if not sentence_words_arabic:
        return "لا يوجد كلمات مكتشفة."
    
    current_sentence = ' '.join(sentence_words_arabic)
    
    # Return cached result if sentence is the same
    if current_sentence == last_processed_sentence and last_result:
        return last_result
    
    # Prevent multiple simultaneous processing
    if processing_lock:
        return last_result if last_result else "جاري المعالجة..."
    
    # Choose meaningfulness check method
    meaningfulness_checks = {
        "similarity": lambda words: is_meaningful_by_similarity(words, similarity_threshold=12.0),
        "keywords": lambda words: is_meaningful_by_length_keywords(words, min_words=3),
        "count_only": lambda words: is_meaningful_by_count_only(words, min_words=4),
        "hybrid": is_meaningful_hybrid,
        "none": lambda words: len(words) >= 2  # No meaningfulness check
    }
    
    check_function = meaningfulness_checks.get(meaningfulness_method, meaningfulness_checks["similarity"])
    
    # Check if sentence is meaningful
    if not check_function(sentence_words_arabic):
        return "يرجى إضافة المزيد من الكلمات لتكوين سؤال مفهوم..."
    
    try:
        processing_lock = True
        
        # Simple question formation
        if current_sentence in reformulation_cache:
            question = reformulation_cache[current_sentence]
            print("⚡ Used cached reformulated question")
        else:
            question = reformulate_to_question(current_sentence)
            reformulation_cache[current_sentence] = question
            print("✨ Reformulated via GPT")

        print("❓ Question formed:", question)
        
        # Search in knowledge base
        input_embedding = model.encode([question])
        D, I = index.search(np.array(input_embedding).astype(np.float32), k=1)
        distance = D[0][0]
        
        print("📏 Knowledge base similarity:", distance)
        
        if distance > threshold:
            print("📉 No close match in QA base, using GPT fallback...")

            if question in answer_cache:
                gpt_fallback_answer = answer_cache[question]
                print("⚡ Used cached GPT answer")
            else:
                start_time = time.time()
                gpt_fallback_answer = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "أنت مساعد طبي تقدم إجابات فقط للأسئلة الصحية."},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.4
                ).choices[0].message.content.strip()

                duration = time.time() - start_time
                print(f"⏱️ GPT response time: {duration:.2f} seconds")

                # Cache the GPT answer
                answer_cache[question] = gpt_fallback_answer

                # Save unmatched questions for future analysis
                with open("unmatched_questions.txt", "a", encoding="utf-8") as f:
                    f.write(question + "\n")

            result = {
                "question": question,
                "answer": gpt_fallback_answer + " (إجابة مولدة من الذكاء الاصطناعي)"
            }

        else:
            context = answers[I[0][0]]
            result = {
                "question": question,
                "answer": context
            }

        # Cache the result
        last_processed_sentence = current_sentence
        last_result = result
        
        print("🧠 Final Answer:", result["answer"])
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "question": "خطأ في المعالجة",
            "answer": "حدث خطأ في معالجة السؤال، يرجى المحاولة مرة أخرى."
        }
    finally:
        processing_lock = False

# Clear cache function
def clear_cache():
    global answer_cache, last_processed_sentence, last_result
    answer_cache.clear()
    last_processed_sentence = ""
    last_result = {}
    print("🧹 Cache cleared")