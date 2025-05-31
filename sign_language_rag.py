#sign_language_rag.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
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

predefined_questions = {
    'تحليل': "هل التحاليل تحتاج صيام؟",
    'دواء': "أين أصرف الدواء في مستشفى الملك سلمان؟",
    'موعد': "كيف أحجز موعد في مستشفى الملك سلمان؟",
    'نتيجة': "أين أستلم نتيجة التحاليل في مستشفى الملك سلمان؟",
    'شكوى': "أريد تقديم شكوى في مستشفى الملك سلمان، كيف؟",
    'صرف': "هل أستطيع استشارة عن الدواء؟",  # Reuse this for now
    'طوارئ': "في حالة الطوارئ، ماذا أفعل؟",  # Closest match
    'صيام': "متى أبدأ الصيام قبل التحليل؟"    # Closest match
}

composite_questions = {
    frozenset(['تحليل', 'صيام']): "هل كل التحاليل تتطلب الصيام؟",
    frozenset(['موعد', 'نتيجة']): "هل أحتاج موعدًا لاستلام نتيجة التحليل؟",
    frozenset(['شكوى', 'طوارئ']): "كيف أقدم شكوى على خدمة الطوارئ؟",
    frozenset(['دواء', 'صرف']): "هل يمكن صرف الدواء من صيدلية خارجية؟",
    frozenset(['تحليل', 'نتيجة']): "كم تستغرق نتيجة التحاليل؟",
    frozenset(['موعد', 'شكوى']): "هل يمكن حجز موعد لتقديم شكوى؟",
    frozenset(['دواء', 'صيام']): "هل الصيام يؤثر على فعالية الدواء؟",
    frozenset(['طوارئ', 'صرف']): "هل يمكن صرف الدواء من الطوارئ؟"
}

triple_questions = {
    frozenset(['تحليل', 'صيام', 'نتيجة']): "إذا صمت قبل التحليل، متى أحصل على النتيجة؟",
    frozenset(['شكوى', 'موعد', 'طوارئ']): "كيف أقدم شكوى بعد زيارة الطوارئ دون موعد؟",
    frozenset(['تحليل', 'نتيجة', 'دواء']): "هل أحتاج نتيجة التحليل لصرف الدواء؟",
    frozenset(['تحليل', 'موعد', 'طوارئ']): "هل يمكن إجراء تحليل في قسم الطوارئ بدون موعد؟",
    frozenset(['دواء', 'شكوى', 'صرف']): "إذا لم يُصرف لي الدواء، كيف أقدم شكوى؟"
}

# Correct Arabic input using GPT-4o
def correct_text(text):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "أنت مصحح لغوي محترف. صحح الجملة التالية نحوياً وإملائياً فقط دون تغيير معناها."},
            {"role": "user", "content": text}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# Generate final answer using retrieved context
def generate_answer(question, context):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "أنت مساعد افتراضي متخصص في تقديم المعلومات الصحية داخل المستشفى. استخدم فقط المعلومة التالية للإجابة على السؤال. إذا لم تجد علاقة واضحة بين المعلومة والسؤال، أجب بـ: لا أعلم. لا تخترع إجابة أبداً."},
            {"role": "user", "content": f"السؤال: {question}\nالمعلومة: {context}"}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# Reformulate input into a clear question (if needed)
def reformulate_to_question(sentence):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "أنت مساعد ذكي في مستشفى. وظيفتك هي تحويل الجمل إلى أسئلة واضحة فقط إذا كانت تتعلق بالمستشفى أو العلاج أو المريض. إذا لم تكن الجملة مرتبطة، أجب فقط بكلمة: غير متعلق بالمستشفى."},
            {"role": "user", "content": sentence}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# Detected words from YOLO mapped to Arabic
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

# 🔑 MAIN callable function for your Flask app
def get_llm_answer(detected_word_en, sentence_words_arabic, threshold=10.0):
    detected_word_ar = translation_dict.get(detected_word_en.lower(), "")
    if not detected_word_ar and sentence_words_arabic:
        detected_word_ar = sentence_words_arabic[-1]

    detected_word_ar = detected_word_ar.strip()

    question = None
    unique_words = set(sentence_words_arabic)

    for combo, q in triple_questions.items():
        if combo.issubset(unique_words):
            question = q
            break

    if not question:
        for combo, q in composite_questions.items():
            if combo.issubset(unique_words):
                question = q
                break

    if not question:
        question = predefined_questions.get(detected_word_ar)

    if not question:
        for word in reversed(sentence_words_arabic):
            if word in predefined_questions:
                question = predefined_questions[word]
                break

    if not question:
        return "لا يمكن التعرف على هذه الكلمة."

    print("❓ Predefined question:", question)

    input_embedding = model.encode([question])
    D, I = index.search(np.array(input_embedding).astype(np.float32), k=1)
    distance = D[0][0]
    print("📏 Similarity Distance:", distance)

    if distance > threshold:
        print("❌ No relevant match found.")
        return "عذرًا، لا أملك معلومات دقيقة عن هذا الاستفسار."

    top_question = questions[I[0][0]]
    context = answers[I[0][0]]
    print("🔍 Top matched question:", top_question)
    print("📚 Retrieved answer:", context)

    response = generate_answer(question, context)
    print("🧠 Final GPT Answer:", response)

    return {
        "question": question,
        "answer": response
    }
