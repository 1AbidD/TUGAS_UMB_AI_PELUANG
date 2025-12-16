"""
logic_prob.py
Core probabilistic logic engine for card-based probability problems.
All calculations are exact and explainable.
"""

from fractions import Fraction
from collections import Counter
import math
import matplotlib.pyplot as plt


# =====================================================
# DOMAIN FACTS — KARTU REMI
# =====================================================

DOMAIN_FACTS = {
    "total_cards": 52,
    "suits": ["heart", "diamond", "club", "spade"],
    "ranks": [
        "ace", "2", "3", "4", "5", "6", "7",
        "8", "9", "10", "jack", "queen", "king"
    ],
    "cards_per_suit": 13,
    "cards_per_rank": 4,
    "red_suits": ["heart", "diamond"],
    "black_suits": ["club", "spade"],
    "face_cards": ["jack", "queen", "king"]
}


# =====================================================
# BASIC ARITHMETIC HELPERS
# =====================================================

def tambah(a, b):
    """Penjumlahan"""
    return a + b

def kurang(a, b):
    """Pengurangan"""
    return a - b

def kali(a, b):
    """Perkalian"""
    return a * b

def bagi(a, b):
    """Pembagian"""
    if b == 0:
        raise ValueError("Tidak bisa dibagi dengan 0")
    return a / b


# =====================================================
# CORE OUTPUT TEMPLATE
# =====================================================

def result_template(
    topic,
    events,
    event_structure,
    sample_space,
    counts,
    formal_steps,
    calculation,
    result
):
    """Template standar untuk hasil perhitungan probabilitas"""
    return {
        "meta": {
            "domain": "kartu_remi",
            "topic": topic
        },
        "domain_facts": DOMAIN_FACTS,
        "events": events,
        "event_structure": event_structure,
        "sample_space": sample_space,
        "counts": counts,
        "formal_steps": formal_steps,
        "calculation": calculation,
        "result": result
    }


# =====================================================
# OPERASI DASAR: DAN, ATAU, BUKAN
# =====================================================

def prob_dan(n_a_and_b, total):
    """
    Peluang kejadian A DAN B (A ∩ B)
    P(A ∩ B) = jumlah(A ∩ B) / total
    
    Args:
        n_a_and_b: jumlah kejadian yang merupakan A DAN B bersamaan (intersection)
        total: total seluruh kejadian yang mungkin
    """
    prob = Fraction(n_a_and_b, total)
    
    return result_template(
        topic="peluang_dan",
        events={
            "A": "kejadian pertama",
            "B": "kejadian kedua"
        },
        event_structure={
            "type": "intersection",
            "expression": "P(A ∩ B)",
            "description": "peluang A dan B terjadi bersamaan (irisan kedua kejadian)"
        },
        sample_space={
            "description": "semua kemungkinan kartu",
            "total_outcomes": total
        },
        counts={
            "A_and_B": n_a_and_b,
            "total": total
        },
        formal_steps=[
            "Rumus: P(A ∩ B) = jumlah(A ∩ B) / total",
            f"Jumlah kartu yang merupakan A DAN B = {n_a_and_b}",
            f"Total kartu = {total}",
            f"Maka: P(A ∩ B) = {n_a_and_b}/{total}"
        ],
        calculation={
            "expression": f"{n_a_and_b} / {total}",
            "simplified": str(prob)
        },
        result={
            "fraction": str(prob),
            "decimal": float(prob),
            "percentage": f"{float(prob) * 100:.2f}%"
        }
    )


def prob_atau(n_a, n_b, n_a_and_b, total):
    """
    Peluang kejadian A ATAU B (A ∪ B)
    P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
    
    Karena ada kartu yang termasuk A dan juga termasuk B (irisan),
    kita kurangi agar tidak dihitung 2x.
    
    Args:
        n_a: jumlah kejadian A
        n_b: jumlah kejadian B
        n_a_and_b: jumlah kejadian yang merupakan A DAN B (irisan)
        total: total seluruh kejadian yang mungkin
    """
    n_a_or_b = n_a + n_b - n_a_and_b
    prob = Fraction(n_a_or_b, total)
    
    return result_template(
        topic="peluang_atau",
        events={
            "A": "kejadian pertama",
            "B": "kejadian kedua"
        },
        event_structure={
            "type": "union",
            "expression": "P(A ∪ B)",
            "description": "peluang A atau B atau keduanya terjadi (gabungan dua kejadian)"
        },
        sample_space={
            "description": "semua kemungkinan kartu",
            "total_outcomes": total
        },
        counts={
            "A": n_a,
            "B": n_b,
            "A_and_B": n_a_and_b,
            "A_or_B": n_a_or_b,
            "total": total
        },
        formal_steps=[
            "Rumus: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)",
            f"Karena ada {n_a_and_b} kartu yang termasuk A DAN B (irisan),",
            f"kita kurangi agar tidak dihitung 2 kali.",
            f"Jumlah A = {n_a}",
            f"Jumlah B = {n_b}",
            f"Jumlah A ∩ B (irisan) = {n_a_and_b}",
            f"Jumlah A ∪ B = {n_a} + {n_b} - {n_a_and_b} = {n_a_or_b}",
            f"Maka: P(A ∪ B) = {n_a_or_b}/{total}"
        ],
        calculation={
            "expression": f"({n_a} + {n_b} - {n_a_and_b}) / {total}",
            "simplified": str(prob)
        },
        result={
            "fraction": str(prob),
            "decimal": float(prob),
            "percentage": f"{float(prob) * 100:.2f}%"
        }
    )


def prob_bukan(n_a, total):
    """
    Peluang BUKAN kejadian A (A') - Complement
    P(A') = 1 - P(A) = (total - n_a) / total
    
    Args:
        n_a: jumlah kejadian A
        total: total seluruh kejadian yang mungkin
    """
    n_not_a = total - n_a
    prob = Fraction(n_not_a, total)
    
    return result_template(
        topic="peluang_bukan",
        events={
            "A": "kejadian yang tidak diinginkan"
        },
        event_structure={
            "type": "complement",
            "expression": "P(A')",
            "description": "peluang A tidak terjadi (komplemen dari A)"
        },
        sample_space={
            "description": "semua kemungkinan kartu",
            "total_outcomes": total
        },
        counts={
            "A": n_a,
            "not_A": n_not_a,
            "total": total
        },
        formal_steps=[
            "Rumus: P(A') = 1 - P(A)",
            "Atau: P(A') = (total - A) / total",
            f"Jumlah A = {n_a}",
            f"Jumlah bukan A = {total} - {n_a} = {n_not_a}",
            f"Maka: P(A') = {n_not_a}/{total}"
        ],
        calculation={
            "expression": f"{n_not_a} / {total}",
            "simplified": str(prob)
        },
        result={
            "fraction": str(prob),
            "decimal": float(prob),
            "percentage": f"{float(prob) * 100:.2f}%"
        }
    )


# =====================================================
# 1. PELUANG BERSYARAT
# =====================================================

def conditional_probability(n_a_and_b, n_b, total):
    """
    Peluang bersyarat P(A | B) - Conditional Probability
    P(A | B) = P(A ∩ B) / P(B) = jumlah(A ∩ B) / jumlah(B)
    
    Artinya: Peluang A terjadi jika kita tahu bahwa B sudah terjadi.
    Ruang sampel berkurang menjadi hanya B, bukan seluruh 52 kartu.
    
    Args:
        n_a_and_b: jumlah kejadian yang merupakan A DAN B (irisan)
        n_b: jumlah kejadian B
        total: total seluruh kejadian (untuk referensi saja)
    """
    if n_b == 0:
        raise ValueError("Kejadian B tidak mungkin terjadi (jumlah B = 0)")
    
    prob = Fraction(n_a_and_b, n_b)
    p_a_and_b = Fraction(n_a_and_b, total)
    p_b = Fraction(n_b, total)

    return result_template(
        topic="peluang_bersyarat",
        events={
            "A": "kejadian yang ditanyakan",
            "B": "kejadian yang diketahui (kondisi)"
        },
        event_structure={
            "type": "conditional",
            "expression": "P(A | B)",
            "description": "peluang A terjadi JIKA B sudah terjadi (diberi syarat B)"
        },
        sample_space={
            "description": "ruang sampel terbatas pada kejadian B saja",
            "total_outcomes": n_b,
            "original_total": total
        },
        counts={
            "A_and_B": n_a_and_b,
            "B": n_b,
            "total": total
        },
        formal_steps=[
            "Rumus: P(A | B) = jumlah(A ∩ B) / jumlah(B)",
            f"(Ruang sampel berkurang menjadi hanya {n_b} kartu yang memenuhi B)",
            f"Jumlah kartu yang merupakan A DAN B = {n_a_and_b}",
            f"Jumlah kartu B = {n_b}",
            f"Maka: P(A | B) = {n_a_and_b}/{n_b}"
        ],
        calculation={
            "expression": f"{n_a_and_b} / {n_b}",
            "simplified": str(prob)
        },
        result={
            "fraction": str(prob),
            "decimal": float(prob),
            "percentage": f"{float(prob) * 100:.2f}%"
        }
    )


# =====================================================
# 2. ATURAN PERKALIAN
# =====================================================

def multiplication_rule(n_a, n_b_given_a, total, total_after_a):
    """
    Aturan perkalian untuk kejadian berurutan
    P(A ∩ B) = P(A) × P(B | A)
    
    Args:
        n_a: jumlah kejadian A
        n_b_given_a: jumlah kejadian B setelah A terjadi
        total: total kartu awal
        total_after_a: total kartu setelah A terjadi
    """
    p_a = Fraction(n_a, total)
    p_b_given_a = Fraction(n_b_given_a, total_after_a)
    p_a_and_b = p_a * p_b_given_a

    return {
        "topic": "aturan_perkalian",
        "events": {
            "A": "kejadian pertama",
            "B": "kejadian kedua (setelah A)"
        },
        "formal_rule": "P(A ∩ B) = P(A) × P(B | A)",
        "steps": [
            f"P(A) = {n_a}/{total} = {str(p_a)}",
            f"P(B | A) = {n_b_given_a}/{total_after_a} = {str(p_b_given_a)}",
            f"P(A ∩ B) = {str(p_a)} × {str(p_b_given_a)} = {str(p_a_and_b)}"
        ],
        "calculation": {
            "P(A)": str(p_a),
            "P(B|A)": str(p_b_given_a),
            "expression": f"{str(p_a)} × {str(p_b_given_a)}",
            "result": str(p_a_and_b)
        },
        "result": {
            "fraction": str(p_a_and_b),
            "decimal": float(p_a_and_b),
            "percentage": f"{float(p_a_and_b) * 100:.2f}%"
        }
    }


# =====================================================
# 3. TEOREMA BAYES
# =====================================================

def bayes_theorem(n_b_and_a, n_a, n_b, total):
    """
    Teorema Bayes
    P(A | B) = [P(B | A) × P(A)] / P(B)
    
    Args:
        n_b_and_a: jumlah B dan A (sama dengan A dan B)
        n_a: jumlah kejadian A
        n_b: jumlah kejadian B
        total: total kartu
    """
    p_b_given_a = Fraction(n_b_and_a, n_a) if n_a > 0 else Fraction(0, 1)
    p_a = Fraction(n_a, total)
    p_b = Fraction(n_b, total)
    
    if n_b == 0:
        raise ValueError("P(B) tidak boleh 0")
    
    numerator = p_b_given_a * p_a
    p_a_given_b = numerator / p_b

    return {
        "topic": "teorema_bayes",
        "events": {
            "A": "kejadian yang dicari peluangnya",
            "B": "kejadian yang diamati/diketahui"
        },
        "formal_rule": "P(A | B) = [P(B | A) × P(A)] / P(B)",
        "steps": [
            f"P(B | A) = {n_b_and_a}/{n_a} = {str(p_b_given_a)}",
            f"P(A) = {n_a}/{total} = {str(p_a)}",
            f"P(B) = {n_b}/{total} = {str(p_b)}",
            f"P(A | B) = [{str(p_b_given_a)} × {str(p_a)}] / {str(p_b)}",
            f"P(A | B) = {str(numerator)} / {str(p_b)} = {str(p_a_given_b)}"
        ],
        "calculation": {
            "P(B|A)": str(p_b_given_a),
            "P(A)": str(p_a),
            "P(B)": str(p_b),
            "numerator": str(numerator),
            "expression": f"({str(p_b_given_a)} × {str(p_a)}) / {str(p_b)}",
            "result": str(p_a_given_b)
        },
        "result": {
            "fraction": str(p_a_given_b),
            "decimal": float(p_a_given_b),
            "percentage": f"{float(p_a_given_b) * 100:.2f}%"
        }
    }


# =====================================================
# 4. FUNGSI PADAT PELUANG (DISKRIT / PMF)
# =====================================================

def pmf_card_rank():
    """
    Probability Mass Function untuk nilai kartu (rank)
    Setiap rank memiliki peluang yang sama: 4/52 = 1/13
    """
    pmf = {}
    for rank in DOMAIN_FACTS["ranks"]:
        pmf[rank] = Fraction(DOMAIN_FACTS["cards_per_rank"], 
                            DOMAIN_FACTS["total_cards"])

    return {
        "topic": "fungsi_padat_peluang",
        "type": "diskrit",
        "variable": "X = nilai kartu (rank)",
        "pmf": {k: str(v) for k, v in pmf.items()},
        "pmf_decimal": {k: float(v) for k, v in pmf.items()},
        "properties": {
            "sum": str(sum(pmf.values())),
            "sum_equals_1": sum(pmf.values()) == 1
        }
    }


def pmf_card_suit():
    """
    Probability Mass Function untuk jenis kartu (suit)
    Setiap suit memiliki peluang: 13/52 = 1/4
    """
    pmf = {}
    for suit in DOMAIN_FACTS["suits"]:
        pmf[suit] = Fraction(DOMAIN_FACTS["cards_per_suit"], 
                            DOMAIN_FACTS["total_cards"])

    return {
        "topic": "fungsi_padat_peluang",
        "type": "diskrit",
        "variable": "Y = jenis kartu (suit)",
        "pmf": {k: str(v) for k, v in pmf.items()},
        "pmf_decimal": {k: float(v) for k, v in pmf.items()},
        "properties": {
            "sum": str(sum(pmf.values())),
            "sum_equals_1": sum(pmf.values()) == 1
        }
    }


# =====================================================
# 5. DISTRIBUSI EMPIRIS
# =====================================================

def empirical_distribution(observations):
    """
    Distribusi empiris dari data observasi
    
    Args:
        observations: list nilai yang diamati
    """
    total = len(observations)
    if total == 0:
        raise ValueError("Tidak ada observasi")
    
    counter = Counter(observations)
    empirical = {k: Fraction(v, total) for k, v in counter.items()}

    return {
        "topic": "distribusi_empiris",
        "observations": total,
        "unique_values": len(counter),
        "distribution": {k: str(v) for k, v in empirical.items()},
        "distribution_decimal": {k: float(v) for k, v in empirical.items()},
        "frequency": dict(counter),
        "properties": {
            "sum": str(sum(empirical.values())),
            "sum_equals_1": sum(empirical.values()) == 1
        }
    }


# =====================================================
# 6. DISTRIBUSI PELUANG GABUNGAN
# =====================================================

def joint_distribution(n_xy, total):
    """
    Distribusi peluang gabungan P(X, Y)
    
    Args:
        n_xy: jumlah kartu dengan karakteristik X dan Y
        total: total kartu
    """
    prob = Fraction(n_xy, total)

    return {
        "topic": "distribusi_gabungan",
        "type": "P(X, Y)",
        "description": "peluang X dan Y terjadi bersamaan",
        "counts": {
            "n(X,Y)": n_xy,
            "total": total
        },
        "calculation": {
            "expression": f"{n_xy} / {total}",
            "result": str(prob)
        },
        "result": {
            "fraction": str(prob),
            "decimal": float(prob),
            "percentage": f"{float(prob) * 100:.2f}%"
        }
    }


def joint_pmf_rank_suit():
    """
    Distribusi gabungan untuk rank dan suit
    P(X=rank, Y=suit) = 1/52 untuk setiap kombinasi
    """
    joint_pmf = {}
    prob = Fraction(1, 52)
    
    for rank in DOMAIN_FACTS["ranks"]:
        for suit in DOMAIN_FACTS["suits"]:
            joint_pmf[f"{rank}_{suit}"] = prob

    return {
        "topic": "distribusi_gabungan",
        "type": "P(Rank, Suit)",
        "description": "distribusi gabungan rank dan suit",
        "total_combinations": len(joint_pmf),
        "probability_per_card": str(prob),
        "sample": {k: str(v) for k, v in list(joint_pmf.items())[:5]},
        "properties": {
            "all_equal": all(v == prob for v in joint_pmf.values()),
            "sum": str(sum(joint_pmf.values())),
            "sum_equals_1": sum(joint_pmf.values()) == 1
        }
    }


# =====================================================
# 7. X DAN Y KEDUANYA KONTINU (KONSEPTUAL)
# =====================================================

def continuous_joint_distribution():
    """
    Model konseptual untuk distribusi gabungan kontinu
    (Tidak berlaku langsung untuk kartu diskrit)
    """
    return {
        "topic": "distribusi_gabungan_kontinu",
        "type": "konseptual",
        "description": "Model untuk variabel kontinu X dan Y",
        "density_function": "f(x, y)",
        "example": {
            "uniform": "f(x, y) = 1, untuk 0 ≤ x ≤ 1, 0 ≤ y ≤ 1",
            "property": "∫∫ f(x,y) dx dy = 1"
        },
        "note": "Kartu remi adalah variabel diskrit, bukan kontinu. Ini hanya model konseptual."
    }


# =====================================================
# 8. DISTRIBUSI MARGINAL
# =====================================================

def marginal_distribution_rank():
    """
    Distribusi marginal untuk rank (mengabaikan suit)
    P(X=rank) = Σ P(X=rank, Y=suit) untuk semua suit
    """
    marginal = {}
    for rank in DOMAIN_FACTS["ranks"]:
        # 4 kartu untuk setiap rank (satu per suit)
        marginal[rank] = Fraction(4, 52)
    
    return {
        "topic": "distribusi_marginal",
        "variable": "X (Rank)",
        "method": "menjumlahkan distribusi gabungan terhadap Y (Suit)",
        "formula": "P(X) = Σ P(X,Y) untuk semua Y",
        "distribution": {k: str(v) for k, v in marginal.items()},
        "distribution_decimal": {k: float(v) for k, v in marginal.items()},
        "properties": {
            "sum": str(sum(marginal.values())),
            "sum_equals_1": sum(marginal.values()) == 1
        }
    }


def marginal_distribution_suit():
    """
    Distribusi marginal untuk suit (mengabaikan rank)
    P(Y=suit) = Σ P(X=rank, Y=suit) untuk semua rank
    """
    marginal = {}
    for suit in DOMAIN_FACTS["suits"]:
        # 13 kartu untuk setiap suit (satu per rank)
        marginal[suit] = Fraction(13, 52)
    
    return {
        "topic": "distribusi_marginal",
        "variable": "Y (Suit)",
        "method": "menjumlahkan distribusi gabungan terhadap X (Rank)",
        "formula": "P(Y) = Σ P(X,Y) untuk semua X",
        "distribution": {k: str(v) for k, v in marginal.items()},
        "distribution_decimal": {k: float(v) for k, v in marginal.items()},
        "properties": {
            "sum": str(sum(marginal.values())),
            "sum_equals_1": sum(marginal.values()) == 1
        }
    }


# =====================================================
# 9. DISTRIBUSI KUMULATIF (CDF)
# =====================================================

def cdf_card_rank(rank_name):
    """
    Cumulative Distribution Function untuk rank kartu
    P(X ≤ rank) = probabilitas mendapat rank ini atau lebih rendah
    
    Args:
        rank_name: nama rank (misalnya "5", "jack", dll)
    """
    if rank_name not in DOMAIN_FACTS["ranks"]:
        raise ValueError(f"Rank tidak valid: {rank_name}")
    
    rank_index = DOMAIN_FACTS["ranks"].index(rank_name)
    # Jumlah rank dari ace sampai rank_name
    num_ranks = rank_index + 1
    # Setiap rank ada 4 kartu
    favorable = num_ranks * 4
    total = 52
    
    prob = Fraction(favorable, total)
    
    return {
        "topic": "distribusi_kumulatif",
        "type": "CDF",
        "variable": "X (Rank)",
        "expression": f"P(X ≤ {rank_name})",
        "description": f"peluang mendapat {rank_name} atau lebih rendah",
        "calculation": {
            "ranks_included": DOMAIN_FACTS["ranks"][:rank_index + 1],
            "num_ranks": num_ranks,
            "cards_per_rank": 4,
            "favorable_cards": favorable,
            "total_cards": total,
            "expression": f"{favorable} / {total}"
        },
        "result": {
            "fraction": str(prob),
            "decimal": float(prob),
            "percentage": f"{float(prob) * 100:.2f}%"
        }
    }


def cdf_all_ranks():
    """
    CDF untuk semua rank kartu
    """
    cdf_values = {}
    cumulative = 0
    
    for rank in DOMAIN_FACTS["ranks"]:
        cumulative += Fraction(4, 52)
        cdf_values[rank] = cumulative
    
    return {
        "topic": "distribusi_kumulatif",
        "type": "CDF lengkap",
        "variable": "X (Rank)",
        "cdf": {k: str(v) for k, v in cdf_values.items()},
        "cdf_decimal": {k: float(v) for k, v in cdf_values.items()},
        "properties": {
            "cdf_at_max": str(cdf_values[DOMAIN_FACTS["ranks"][-1]]),
            "cdf_max_equals_1": cdf_values[DOMAIN_FACTS["ranks"][-1]] == 1
        }
    }


# =====================================================
# 10. GRAFIK DISTRIBUSI PELUANG
# =====================================================

def plot_pmf_rank(save_path=None):
    """
    Membuat grafik PMF untuk rank kartu
    """
    ranks = DOMAIN_FACTS["ranks"]
    probabilities = [float(Fraction(4, 52))] * 13

    plt.figure(figsize=(12, 6))
    plt.bar(ranks, probabilities, color='steelblue', edgecolor='black')
    plt.xlabel("Nilai Kartu (Rank)", fontsize=12)
    plt.ylabel("Peluang", fontsize=12)
    plt.title("Distribusi Peluang Nilai Kartu (PMF)", fontsize=14, fontweight='bold')
    plt.ylim(0, max(probabilities) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    
    # Tambahkan label nilai pada bar
    for i, (rank, prob) in enumerate(zip(ranks, probabilities)):
        plt.text(i, prob + 0.002, f'{prob:.4f}', ha='center', va='bottom', fontsize=8)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    
    return {
        "topic": "grafik_pmf",
        "type": "bar_chart",
        "variable": "Rank",
        "status": "created"
    }


def plot_cdf_rank(save_path=None):
    """
    Membuat grafik CDF untuk rank kartu
    """
    ranks = DOMAIN_FACTS["ranks"]
    cdf_data = cdf_all_ranks()
    cdf_values = [float(cdf_data["cdf_decimal"][rank]) for rank in ranks]

    plt.figure(figsize=(12, 6))
    plt.step(range(len(ranks)), cdf_values, where='post', linewidth=2, color='darkred')
    plt.scatter(range(len(ranks)), cdf_values, color='darkred', s=50, zorder=5)
    plt.xticks(range(len(ranks)), ranks)
    plt.xlabel("Nilai Kartu (Rank)", fontsize=12)
    plt.ylabel("Peluang Kumulatif", fontsize=12)
    plt.title("Distribusi Kumulatif Nilai Kartu (CDF)", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    
    return {
        "topic": "grafik_cdf",
        "type": "step_chart",
        "variable": "Rank",
        "status": "created"
    }


def plot_suit_distribution(save_path=None):
    """
    Membuat grafik distribusi untuk suit
    """
    suits = DOMAIN_FACTS["suits"]
    probabilities = [float(Fraction(13, 52))] * 4
    colors = ['red', 'red', 'black', 'black']

    plt.figure(figsize=(8, 6))
    plt.bar(suits, probabilities, color=colors, edgecolor='black', alpha=0.7)
    plt.xlabel("Jenis Kartu (Suit)", fontsize=12)
    plt.ylabel("Peluang", fontsize=12)
    plt.title("Distribusi Peluang Jenis Kartu (Suit)", fontsize=14, fontweight='bold')
    plt.ylim(0, max(probabilities) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    
    # Tambahkan label nilai pada bar
    for i, (suit, prob) in enumerate(zip(suits, probabilities)):
        plt.text(i, prob + 0.005, f'{prob:.4f}', ha='center', va='bottom', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    
    return {
        "topic": "grafik_suit",
        "type": "bar_chart",
        "variable": "Suit",
        "status": "created"
    }


# =====================================================
# 11. EXPLAIN PROBABILITY
# =====================================================

def explain_probability(result: dict, include_result=False) -> str:
    """
    Mengubah hasil perhitungan probabilitas menjadi penjelasan teks
    yang siap dijelaskan oleh AI.
    
    Args:
        result: dict hasil dari fungsi probability
        include_result: jika False, tidak menampilkan "HASIL AKHIR:" di akhir
    """
    explanation = []

    # Events
    if "events" in result:
        explanation.append("KEJADIAN:")
        for name, desc in result["events"].items():
            explanation.append(f"  {name}: {desc}")
        explanation.append("")

    # Event Structure
    if "event_structure" in result:
        struct = result["event_structure"]
        explanation.append(f"TIPE: {struct.get('type', 'N/A')}")
        explanation.append(f"EKSPRESI: {struct.get('expression', 'N/A')}")
        explanation.append(f"DESKRIPSI: {struct.get('description', 'N/A')}")
        explanation.append("")

    # Sample Space
    if "sample_space" in result:
        space = result["sample_space"]
        explanation.append("RUANG SAMPEL:")
        explanation.append(f"  {space.get('description', 'N/A')}")
        explanation.append(f"  Total outcomes: {space.get('total_outcomes', 'N/A')}")
        explanation.append("")

    # Counts
    if "counts" in result:
        explanation.append("JUMLAH:")
        for key, value in result["counts"].items():
            explanation.append(f"  {key} = {value}")
        explanation.append("")

    # Formal Steps
    if "formal_steps" in result:
        explanation.append("LANGKAH FORMAL:")
        for i, step in enumerate(result["formal_steps"], 1):
            explanation.append(f"  {i}. {step}")
        explanation.append("")

    # Steps (alternative)
    if "steps" in result:
        explanation.append("LANGKAH PERHITUNGAN:")
        for i, step in enumerate(result["steps"], 1):
            explanation.append(f"  {i}. {step}")
        explanation.append("")

    # Calculation
    if "calculation" in result:
        explanation.append("PERHITUNGAN:")
        calc = result["calculation"]
        if isinstance(calc, dict):
            for key, value in calc.items():
                explanation.append(f"  {key}: {value}")
        else:
            explanation.append(f"  {calc}")
        explanation.append("")

    # Result (only if include_result=True)
    if include_result and "result" in result:
        explanation.append("HASIL AKHIR:")
        res = result["result"]
        if isinstance(res, dict):
            if "fraction" in res:
                explanation.append(f"  Pecahan: {res['fraction']}")
            if "decimal" in res:
                explanation.append(f"  Desimal: {res['decimal']:.6f}")
            if "percentage" in res:
                explanation.append(f"  Persentase: {res['percentage']}")
        else:
            explanation.append(f"  {res}")

    return "\n".join(explanation)


# =====================================================
# 12. HELPER FUNCTIONS - CARD IDENTIFICATION
# =====================================================

def identify_card_properties(card_name):
    """
    Identifikasi properti kartu berdasarkan namanya
    
    Args:
        card_name: nama kartu (contoh: "ace_heart", "king_spade")
    
    Returns:
        dict: properti kartu
    """
    try:
        parts = card_name.lower().split("_")
        rank = parts[0]
        suit = parts[1] if len(parts) > 1 else None
        
        properties = {
            "rank": rank,
            "suit": suit,
            "is_face_card": rank in DOMAIN_FACTS["face_cards"],
            "is_red": suit in DOMAIN_FACTS["red_suits"] if suit else None,
            "is_black": suit in DOMAIN_FACTS["black_suits"] if suit else None,
            "valid": rank in DOMAIN_FACTS["ranks"] and (suit in DOMAIN_FACTS["suits"] if suit else True)
        }
        
        return properties
    except Exception as e:
        return {"error": str(e), "valid": False}


def count_cards_by_property(property_type, property_value):
    """
    Menghitung jumlah kartu berdasarkan properti tertentu
    
    Args:
        property_type: jenis properti ("rank", "suit", "color", "face")
        property_value: nilai properti
    
    Returns:
        int: jumlah kartu
    """
    if property_type == "rank":
        if property_value in DOMAIN_FACTS["ranks"]:
            return DOMAIN_FACTS["cards_per_rank"]
    
    elif property_type == "suit":
        if property_value in DOMAIN_FACTS["suits"]:
            return DOMAIN_FACTS["cards_per_suit"]
    
    elif property_type == "color":
        if property_value.lower() == "red":
            return len(DOMAIN_FACTS["red_suits"]) * DOMAIN_FACTS["cards_per_suit"]
        elif property_value.lower() == "black":
            return len(DOMAIN_FACTS["black_suits"]) * DOMAIN_FACTS["cards_per_suit"]
    
    elif property_type == "face":
        if property_value.lower() == "face_card":
            return len(DOMAIN_FACTS["face_cards"]) * DOMAIN_FACTS["cards_per_rank"]
        elif property_value.lower() == "non_face":
            non_face_ranks = len(DOMAIN_FACTS["ranks"]) - len(DOMAIN_FACTS["face_cards"])
            return non_face_ranks * DOMAIN_FACTS["cards_per_rank"]
    
    return 0


# =====================================================
# 13. QUICK CALCULATION HELPERS
# =====================================================

def quick_prob_red_card():
    """Peluang mendapat kartu merah"""
    n_red = count_cards_by_property("color", "red")
    return prob_dan(n_red, DOMAIN_FACTS["total_cards"])


def quick_prob_face_card():
    """Peluang mendapat kartu wajah (Jack, Queen, King)"""
    n_face = count_cards_by_property("face", "face_card")
    return prob_dan(n_face, DOMAIN_FACTS["total_cards"])


def quick_prob_red_and_face():
    """Peluang mendapat kartu merah DAN kartu wajah"""
    # Ada 6 kartu: Jack/Queen/King dari Heart dan Diamond
    n_red_face = len(DOMAIN_FACTS["face_cards"]) * len(DOMAIN_FACTS["red_suits"])
    return prob_dan(n_red_face, DOMAIN_FACTS["total_cards"])


def quick_prob_red_or_face():
    """Peluang mendapat kartu merah ATAU kartu wajah"""
    n_red = count_cards_by_property("color", "red")
    n_face = count_cards_by_property("face", "face_card")
    n_red_face = len(DOMAIN_FACTS["face_cards"]) * len(DOMAIN_FACTS["red_suits"])
    return prob_atau(n_red, n_face, n_red_face, DOMAIN_FACTS["total_cards"])


# =====================================================
# 14. EXAMPLE USAGE & TESTING
# =====================================================

def run_examples():
    """
    Menjalankan contoh-contoh perhitungan probabilitas
    """
    print("=" * 60)
    print("CONTOH PERHITUNGAN PROBABILITAS KARTU REMI")
    print("=" * 60)
    
    # Contoh 1: Peluang DAN
    print("\n1. PELUANG KARTU MERAH:")
    result = quick_prob_red_card()
    print(explain_probability(result))
    
    # Contoh 2: Peluang ATAU
    print("\n2. PELUANG KARTU MERAH ATAU KARTU WAJAH:")
    result = quick_prob_red_or_face()
    print(explain_probability(result))
    
    # Contoh 3: Peluang Bersyarat
    print("\n3. PELUANG BERSYARAT - Kartu wajah jika kartu merah:")
    n_red = 26
    n_red_and_face = 6
    result = conditional_probability(n_red_and_face, n_red, 52)
    print(explain_probability(result))
    
    # Contoh 4: PMF Rank
    print("\n4. FUNGSI PADAT PELUANG (PMF) - RANK:")
    result = pmf_card_rank()
    print(f"Topic: {result['topic']}")
    print(f"Variable: {result['variable']}")
    print(f"Sample probabilities: {list(result['pmf'].items())[:3]}")
    
    # Contoh 5: CDF
    print("\n5. DISTRIBUSI KUMULATIF (CDF) - P(X ≤ 5):")
    result = cdf_card_rank("5")
    print(f"Expression: {result['expression']}")
    print(f"Result: {result['result']['fraction']} = {result['result']['percentage']}")
    
    print("\n" + "=" * 60)
    print("CONTOH SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    # Jalankan contoh jika file dijalankan langsung
    run_examples()