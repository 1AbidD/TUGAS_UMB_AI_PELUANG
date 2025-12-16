import requests
import sqlite3
import json
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os

from logic_prob import (
    explain_probability,
    prob_dan, prob_atau, prob_bukan,
    conditional_probability,
    count_cards_by_property,
    DOMAIN_FACTS
)

# ==============================
# CONFIG
# ==============================
DB_PATH = "data/chat_history.db"

LLM_API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

MAX_HISTORY = 4

# Flask App
app = Flask(__name__, static_folder='assets', static_url_path='/assets')
CORS(app)


# ==============================
# DATABASE
# ==============================
def get_db():
    return sqlite3.connect(DB_PATH)


def load_chat_history(limit=MAX_HISTORY):
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT user_input, bot_response
        FROM chat_history
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cur.fetchall()
    conn.close()

    history = []
    for user_input, bot_response in reversed(rows):
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": bot_response})

    return history


def save_chat(user_text, assistant_text):
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO chat_history (user_input, bot_response, created_at)
        VALUES (?, ?, ?)
        """,
        (user_text, assistant_text, datetime.now()),
    )

    conn.commit()
    conn.close()


# ==============================
# PROBABILITY PARSER
# ==============================

def parse_card_entity(text):
    """
    Parse entity kartu dari text
    Returns: (type, value, count)
    type: 'rank', 'suit', 'color', 'face', 'specific_card'
    """
    text = text.lower().strip()
    
    # Specific card (ace of hearts, king of spades, dll)
    specific_pattern = r'(ace|[2-9]|10|jack|queen|king)\s+(of\s+)?(heart|diamond|club|spade)'
    match = re.search(specific_pattern, text)
    if match:
        rank = match.group(1)
        suit = match.group(3)
        return ('specific_card', f"{rank}_{suit}", 1)
    
    # Rank
    if text in DOMAIN_FACTS["ranks"]:
        count = DOMAIN_FACTS["cards_per_rank"]
        return ('rank', text, count)
    
    # Suit
    if text in DOMAIN_FACTS["suits"]:
        count = DOMAIN_FACTS["cards_per_suit"]
        return ('suit', text, count)
    
    # Face card
    if 'face' in text or text in ['jack', 'queen', 'king']:
        count = len(DOMAIN_FACTS["face_cards"]) * DOMAIN_FACTS["cards_per_rank"]
        return ('face', 'face_card', count)
    
    # Color
    if 'red' in text or 'merah' in text:
        count = len(DOMAIN_FACTS["red_suits"]) * DOMAIN_FACTS["cards_per_suit"]
        return ('color', 'red', count)
    
    if 'black' in text or 'hitam' in text:
        count = len(DOMAIN_FACTS["black_suits"]) * DOMAIN_FACTS["cards_per_suit"]
        return ('color', 'black', count)
    
    return (None, None, 0)


def count_intersection(entity_a, entity_b):
    """
    Hitung irisan antara dua entity
    """
    type_a, val_a, count_a = entity_a
    type_b, val_b, count_b = entity_b
    
    # Specific card - tidak ada irisan kecuali sama persis
    if type_a == 'specific_card' and type_b == 'specific_card':
        return 1 if val_a == val_b else 0
    
    # Specific card dengan rank
    if type_a == 'specific_card' and type_b == 'rank':
        rank = val_a.split('_')[0]
        return 1 if rank == val_b else 0
    
    if type_b == 'specific_card' and type_a == 'rank':
        rank = val_b.split('_')[0]
        return 1 if rank == val_a else 0
    
    # Specific card dengan suit
    if type_a == 'specific_card' and type_b == 'suit':
        suit = val_a.split('_')[1]
        return 1 if suit == val_b else 0
    
    if type_b == 'specific_card' and type_a == 'suit':
        suit = val_b.split('_')[1]
        return 1 if suit == val_a else 0
    
    # Rank dan Suit - selalu ada 1 irisan
    if (type_a == 'rank' and type_b == 'suit') or (type_a == 'suit' and type_b == 'rank'):
        return 1
    
    # Rank dan Color
    if type_a == 'rank' and type_b == 'color':
        # Setiap rank ada 4 kartu: 2 merah, 2 hitam
        return 2
    if type_b == 'rank' and type_a == 'color':
        return 2
    
    # Suit dan Color
    if type_a == 'suit' and type_b == 'color':
        if val_a in DOMAIN_FACTS["red_suits"] and val_b == 'red':
            return 13
        if val_a in DOMAIN_FACTS["black_suits"] and val_b == 'black':
            return 13
        return 0
    
    if type_b == 'suit' and type_a == 'color':
        if val_b in DOMAIN_FACTS["red_suits"] and val_a == 'red':
            return 13
        if val_b in DOMAIN_FACTS["black_suits"] and val_a == 'black':
            return 13
        return 0
    
    # Face dan Suit
    if type_a == 'face' and type_b == 'suit':
        return len(DOMAIN_FACTS["face_cards"])  # 3 face cards per suit
    if type_b == 'face' and type_a == 'suit':
        return len(DOMAIN_FACTS["face_cards"])
    
    # Face dan Color
    if type_a == 'face' and type_b == 'color':
        # 3 face cards × 2 suits (per color)
        return len(DOMAIN_FACTS["face_cards"]) * 2
    if type_b == 'face' and type_a == 'color':
        return len(DOMAIN_FACTS["face_cards"]) * 2
    
    # Face dan Rank
    if type_a == 'face' and type_b == 'rank':
        if val_b in DOMAIN_FACTS["face_cards"]:
            return 4
        return 0
    if type_b == 'face' and type_a == 'rank':
        if val_a in DOMAIN_FACTS["face_cards"]:
            return 4
        return 0
    
    # Same type - no intersection unless same value
    if type_a == type_b:
        return count_a if val_a == val_b else 0
    
    return 0


def parse_probability_query(text):
    """
    Parse query probabilitas dari user input
    Returns: dict dengan info lengkap atau None
    """
    text_lower = text.lower()
    
    # Deteksi operator
    is_dan = any(keyword in text_lower for keyword in ['dan', 'and', '&'])
    is_atau = any(keyword in text_lower for keyword in ['atau', 'or', '|'])
    is_bukan = any(keyword in text_lower for keyword in ['bukan', 'not', 'tidak'])
    is_conditional = any(keyword in text_lower for keyword in ['jika', 'given', 'bila', 'kalau', 'jikalau'])
    
    # Split text untuk mendapat entities
    # Hapus kata-kata noise
    clean_text = text_lower
    for noise in ['peluang', 'probabilitas', 'berapa', 'hitung', 'cari', 'probability', 'chance']:
        clean_text = clean_text.replace(noise, '')
    
    # Parse entities
    entities = []
    
    if is_dan:
        parts = re.split(r'\s+dan\s+|\s+and\s+', clean_text, maxsplit=1)
        if len(parts) >= 2:
            entity_a = parse_card_entity(parts[0])
            entity_b = parse_card_entity(parts[1])
            if entity_a[0] and entity_b[0]:
                return {
                    'operation': 'dan',
                    'entity_a': entity_a,
                    'entity_b': entity_b
                }
    
    if is_atau:
        parts = re.split(r'\s+atau\s+|\s+or\s+', clean_text, maxsplit=1)
        if len(parts) >= 2:
            entity_a = parse_card_entity(parts[0])
            entity_b = parse_card_entity(parts[1])
            if entity_a[0] and entity_b[0]:
                return {
                    'operation': 'atau',
                    'entity_a': entity_a,
                    'entity_b': entity_b
                }
    
    if is_bukan:
        # Ambil entity setelah kata "bukan"
        match = re.search(r'bukan\s+(.+)', clean_text)
        if match:
            entity = parse_card_entity(match.group(1))
            if entity[0]:
                return {
                    'operation': 'bukan',
                    'entity_a': entity
                }
    
    if is_conditional:
        # Format: P(A | B) atau "A jika B"
        match = re.search(r'(.+?)\s+(jika|given|bila|kalau)\s+(.+)', clean_text)
        if match:
            entity_a = parse_card_entity(match.group(1))
            entity_b = parse_card_entity(match.group(3))
            if entity_a[0] and entity_b[0]:
                return {
                    'operation': 'conditional',
                    'entity_a': entity_a,
                    'entity_b': entity_b
                }
    
    # Single entity (simple probability)
    entity = parse_card_entity(clean_text)
    if entity[0]:
        return {
            'operation': 'single',
            'entity_a': entity
        }
    
    return None


def calculate_probability(query_info):
    """
    Hitung probabilitas berdasarkan parsed query
    """
    if not query_info:
        return None
    
    operation = query_info['operation']
    total = DOMAIN_FACTS['total_cards']
    
    # SINGLE PROBABILITY
    if operation == 'single':
        entity = query_info['entity_a']
        n = entity[2]
        result = prob_dan(n, total)
        result['context'] = f"Peluang mendapat {entity[1]}"
        return result
    
    # DAN (INTERSECTION)
    elif operation == 'dan':
        entity_a = query_info['entity_a']
        entity_b = query_info['entity_b']
        n_intersection = count_intersection(entity_a, entity_b)
        
        result = prob_dan(n_intersection, total)
        result['context'] = f"Peluang mendapat {entity_a[1]} DAN {entity_b[1]}"
        result['detail'] = {
            'entity_a': {'type': entity_a[0], 'value': entity_a[1], 'count': entity_a[2]},
            'entity_b': {'type': entity_b[0], 'value': entity_b[1], 'count': entity_b[2]},
            'intersection': n_intersection
        }
        return result
    
    # ATAU (UNION)
    elif operation == 'atau':
        entity_a = query_info['entity_a']
        entity_b = query_info['entity_b']
        n_a = entity_a[2]
        n_b = entity_b[2]
        n_intersection = count_intersection(entity_a, entity_b)
        
        result = prob_atau(n_a, n_b, n_intersection, total)
        result['context'] = f"Peluang mendapat {entity_a[1]} ATAU {entity_b[1]}"
        result['detail'] = {
            'entity_a': {'type': entity_a[0], 'value': entity_a[1], 'count': n_a},
            'entity_b': {'type': entity_b[0], 'value': entity_b[1], 'count': n_b},
            'intersection': n_intersection
        }
        return result
    
    # BUKAN (COMPLEMENT)
    elif operation == 'bukan':
        entity = query_info['entity_a']
        n = entity[2]
        
        result = prob_bukan(n, total)
        result['context'] = f"Peluang BUKAN {entity[1]}"
        result['detail'] = {
            'entity': {'type': entity[0], 'value': entity[1], 'count': n}
        }
        return result
    
    # CONDITIONAL P(A|B)
    elif operation == 'conditional':
        entity_a = query_info['entity_a']
        entity_b = query_info['entity_b']
        n_b = entity_b[2]
        n_intersection = count_intersection(entity_a, entity_b)
        
        result = conditional_probability(n_intersection, n_b, total)
        result['context'] = f"Peluang {entity_a[1]} jika {entity_b[1]}"
        result['detail'] = {
            'entity_a': {'type': entity_a[0], 'value': entity_a[1], 'count': entity_a[2]},
            'entity_b': {'type': entity_b[0], 'value': entity_b[1], 'count': n_b},
            'intersection': n_intersection
        }
        return result
    
    return None


# ==============================
# INTENT DETECTION
# ==============================
def is_probability_question(text: str) -> bool:
    """
    Deteksi apakah pertanyaan adalah soal probabilitas yang memerlukan perhitungan.
    Return True hanya jika ada operasi probabilitas yang jelas (dan/atau/bukan/conditional).
    """
    t = text.lower()

    # Keywords yang menunjukkan pertanyaan penjelasan, bukan perhitungan
    EXPLAIN_KEYWORDS = [
        "kenapa", "mengapa", "jelaskan", "alasannya", "why", "explain",
        "apakah kamu bisa", "bisa membedakan", "apa aja"
    ]

    # Jika pertanyaan adalah penjelasan umum, bukan soal perhitungan
    if any(k in t for k in EXPLAIN_KEYWORDS):
        return False

    # Keywords yang menunjukkan soal probabilitas
    PROB_KEYWORDS = [
        "peluang", "probabilitas", "kemungkinan",
        "berapa", "chance", "probability", "hitung"
    ]

    # Harus ada keyword probabilitas
    if not any(k in t for k in PROB_KEYWORDS):
        return False
    
    # Harus ada operasi: dan/atau/bukan/conditional
    OPERATION_KEYWORDS = [
        'dan', 'and', '&',           # AND/Intersection
        'atau', 'or', '|',            # OR/Union
        'bukan', 'not', 'tidak',       # NOT/Complement
        'jika', 'given', 'bila', 'kalau', 'jikalau'  # CONDITIONAL
    ]
    
    has_operation = any(op in t for op in OPERATION_KEYWORDS)
    
    # Hanya return True jika ada prob keyword DAN ada operation keyword
    return has_operation


# ==============================
# PROMPT BUILDER
# ==============================
def build_prompt(user_input, prob_result=None):
    system_prompt = (
        "Kamu adalah AI tutor probabilitas dan logika matematika. "
        "Tugasmu adalah menjelaskan hasil perhitungan probabilitas yang sudah dihitung oleh sistem. "
        "JANGAN menghitung ulang atau mengubah angka. "
        "JANGAN menambahkan penjelasan operasi lain yang tidak ditanyakan. "
        "Jelaskan HANYA apa yang diminta user, dengan bahasa sederhana, runtut, dan logis. "
        "\n\n"
        "Format penjelasan:\n"
        "1. Jelaskan konteks masalah (kejadian apa yang ditanya)\n"
        "2. Jelaskan data yang diketahui (jumlah kartu, kejadian A, B)\n"
        "3. Jelaskan rumus yang digunakan\n"
        "4. Jelaskan langkah perhitungan\n"
        "5. Simpulkan hasil akhir\n"
        "\n"
        "PENTING: Hanya jelaskan operasi yang ada di HASIL PERHITUNGAN SISTEM di bawah.\n"
        "Jangan menambahkan penjelasan operasi lain (seperti union jika ditanya intersection).\n"
    )

    messages = [{"role": "system", "content": system_prompt}]
    
    # Jika ada hasil probabilitas, tambahkan sebagai context
    if prob_result:
        topic = prob_result.get("meta", {}).get("topic", "probabilitas")
        # Jangan include hasil di explain_probability, karena akan ditambah di response
        system_context = explain_probability(prob_result, include_result=False)
        
        context = f"""
HASIL PERHITUNGAN SISTEM:
{system_context}

INSTRUKSI SPESIFIK:
- Jelaskan perhitungan di atas
- Gunakan bahasa yang mudah dipahami mahasiswa
- JANGAN tambahkan penjelasan operasi probabilitas lain
- JANGAN berhitung alternatif atau operasi lain
"""
        messages.append({"role": "system", "content": context})
    
    messages.extend(load_chat_history())
    messages.append({"role": "user", "content": user_input})

    return messages


# ==============================
# LLM STREAMING
# ==============================
def call_llm_stream(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": True,
    }

    full_response = ""

    try:
        with requests.post(
            LLM_API_URL,
            json=payload,
            stream=True,
            timeout=120,
        ) as res:

            if res.status_code != 200:
                raise RuntimeError(f"LLM ERROR {res.status_code}: {res.text}")

            for line in res.iter_lines():
                if not line:
                    continue

                if line.startswith(b"data: "):
                    data = line[len(b"data: "):]

                    if data == b"[DONE]":
                        break

                    try:
                        chunk = json.loads(data.decode("utf-8"))
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            text = delta["content"]
                            # Filter out 'None' string from output
                            if text.strip().lower() != "none":
                                print(text, end="", flush=True)
                                full_response += text
                    except Exception:
                        continue
    except Exception as e:
        return f"❌ Error calling LLM: {str(e)}"

    print()
    response = full_response.strip()
    # Remove any remaining 'None' text
    response = response.replace("None", "").strip()
    return response


def call_llm_stream_generator(messages):
    """Generator version untuk streaming ke frontend"""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": True,
    }

    try:
        with requests.post(
            LLM_API_URL,
            json=payload,
            stream=True,
            timeout=120,
        ) as res:

            if res.status_code != 200:
                raise RuntimeError(f"LLM ERROR {res.status_code}: {res.text}")

            for line in res.iter_lines():
                if not line:
                    continue

                if line.startswith(b"data: "):
                    data = line[len(b"data: "):]

                    if data == b"[DONE]":
                        break

                    try:
                        chunk = json.loads(data.decode("utf-8"))
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            text = delta["content"]
                            # Filter out 'None' string from output
                            if text.strip().lower() != "none":
                                yield text
                    except Exception:
                        continue

    except Exception as e:
        yield f"❌ Error calling LLM: {str(e)}"# ==============================
# MAIN LOOP
# ==============================
def main():
    print("🤖 AI Tutor Probabilitas Kartu Remi")
    print("=" * 50)
    print("Contoh pertanyaan:")
    print("  - Peluang ace dan heart")
    print("  - Berapa peluang kartu merah atau king")
    print("  - Peluang bukan face card")
    print("  - Peluang ace jika kartu merah")
    print("=" * 50)
    print("Ketik 'exit' untuk keluar\n")

    while True:
        user_input = input("Kamu: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("👋 Sampai jumpa.")
            break

        if not user_input:
            continue

        # ============================================================
        # FLOW UTAMA:
        # 1. Deteksi apakah pertanyaan adalah soal PROBABILITAS
        #    (keyword "peluang/berapa" + operasi "dan/atau/bukan/jika")
        # 2a. JIKA YA: Gunakan logic_prob.py untuk hitung,
        #     kemudian LLM jelaskan hasil perhitungan
        # 2b. JIKA TIDAK: LLM gunakan logic sendiri, tanpa logic_prob.py
        # ============================================================

        # 1️⃣ CEK: Apakah ini soal PROBABILITAS?
        if is_probability_question(user_input):
            # 2a️⃣ ROUTE PROBABILITAS - Gunakan logic_prob.py
            query_info = parse_probability_query(user_input)
            
            if query_info:
                prob_result = calculate_probability(query_info)
                
                if prob_result:
                    # Build prompt dengan hasil perhitungan dari logic_prob.py
                    messages = build_prompt(user_input, prob_result)
                    print("\nAI: ", end="", flush=True)
                    explanation = call_llm_stream(messages)
                    
                    # Simpan ke database
                    final_answer = f"📊 {prob_result['context']}\n\n{explanation}\n\n📌 Hasil: {prob_result['result']['fraction']} = {prob_result['result']['percentage']}"
                    save_chat(user_input, final_answer)
                    print()
                    continue
            
            # Jika parsing gagal (operasi terdeteksi tapi entity tidak valid)
            print("\n❌ Maaf, saya belum bisa memahami pertanyaan Anda.")
            print("Coba format seperti: 'peluang ace dan heart' atau 'berapa peluang kartu merah'")
            print()
            continue

        # 2b️⃣ ROUTE CHAT BIASA - LLM gunakan logic sendiri (tanpa logic_prob.py)
        messages = build_prompt(user_input, prob_result=None)

        print("\nAI: ", end="", flush=True)
        response = call_llm_stream(messages)

        save_chat(user_input, response)
        print()


# ==============================
# WEB SERVER ROUTES
# ==============================
@app.route('/')
def index():
    """Serve chatbot UI"""
    return app.send_static_file('chat.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint untuk chat dengan streaming response"""
    from flask import Response
    
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({'error': 'Message is empty'}), 400
        
        # Collect full response for database saving
        full_response_collector = {'text': ''}
        
        # 1️⃣ CEK: Apakah ini soal PROBABILITAS?
        if is_probability_question(user_input):
            # 2a️⃣ ROUTE PROBABILITAS - Gunakan logic_prob.py
            query_info = parse_probability_query(user_input)
            
            if query_info:
                prob_result = calculate_probability(query_info)
                
                if prob_result:
                    # Build prompt dengan hasil perhitungan dari logic_prob.py
                    messages = build_prompt(user_input, prob_result)
                    
                    def generate():
                        # Stream context
                        context_text = f"📊 {prob_result['context']}\n\n"
                        full_response_collector['text'] += context_text
                        yield context_text
                        
                        # Stream LLM explanation
                        explanation_generator = call_llm_stream_generator(messages)
                        for text in explanation_generator:
                            full_response_collector['text'] += text
                            yield text
                        
                        # Stream hasil calculation
                        hasil_text = f"\n\n📌 **Jawaban Akhir:**\n{prob_result['result']['fraction']} = {prob_result['result']['percentage']}"
                        full_response_collector['text'] += hasil_text
                        yield hasil_text
                    
                    response = Response(generate(), mimetype='text/plain')
                    
                    # Save after response finishes
                    @response.call_on_close
                    def save_after_stream():
                        save_chat(user_input, full_response_collector['text'])
                    
                    return response
            
            # Jika parsing gagal
            response_text = "❌ Maaf, saya belum bisa memahami pertanyaan Anda. Coba format seperti: 'peluang ace dan heart' atau 'berapa peluang kartu merah'"
            save_chat(user_input, response_text)
            return Response(response_text, mimetype='text/plain')
        
        # 2b️⃣ ROUTE CHAT BIASA - LLM gunakan logic sendiri (tanpa logic_prob.py)
        messages = build_prompt(user_input, prob_result=None)
        
        def generate():
            explanation_generator = call_llm_stream_generator(messages)
            for text in explanation_generator:
                full_response_collector['text'] += text
                yield text
        
        response = Response(generate(), mimetype='text/plain')
        
        @response.call_on_close
        def save_after_stream():
            save_chat(user_input, full_response_collector['text'])
        
        return response
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


def process_message(user_input):
    """
    Process user message dan return AI response.
    Digunakan oleh baik CLI maupun Web API.
    """
    # 1️⃣ CEK: Apakah ini soal PROBABILITAS?
    if is_probability_question(user_input):
        # 2a️⃣ ROUTE PROBABILITAS - Gunakan logic_prob.py
        query_info = parse_probability_query(user_input)
        
        if query_info:
            prob_result = calculate_probability(query_info)
            
            if prob_result:
                # Build prompt dengan hasil perhitungan dari logic_prob.py
                messages = build_prompt(user_input, prob_result)
                explanation = call_llm_stream(messages)
                
                # Return formatted response
                return f"📊 {prob_result['context']}\n\n{explanation}\n\n📌 Hasil: {prob_result['result']['fraction']} = {prob_result['result']['percentage']}"
        
        # Jika parsing gagal
        return "❌ Maaf, saya belum bisa memahami pertanyaan Anda. Coba format seperti: 'peluang ace dan heart' atau 'berapa peluang kartu merah'"
    
    # 2b️⃣ ROUTE CHAT BIASA - LLM gunakan logic sendiri (tanpa logic_prob.py)
    messages = build_prompt(user_input, prob_result=None)
    response = call_llm_stream(messages)
    
    return response


if __name__ == "__main__":
    import sys
    
    # Check command line argument
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        # Run CLI mode
        main()
    else:
        # Run Web Server mode (default)
        print("🚀 Starting AI Tutor Web Server...")
        print("📱 Open http://localhost:5000 in your browser")
        print("Press Ctrl+C to stop\n")
        app.run(debug=False, host='0.0.0.0', port=5000)