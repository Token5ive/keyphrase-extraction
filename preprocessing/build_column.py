import os
import re
import numpy as np
import datasets


def replace_formulas(text):
    """
    텍스트의 수식을 <formula> 토큰으로 치환 (레벨 2: 균형적)
    
    치환 대상:
        - LaTeX 형식: $...$, $$...$$, \(...\), \[...\]
        - 그리스 문자: α, β, γ, δ, ε, ζ, η, θ, λ, μ, ν, σ, τ, φ, ψ, ω, Δ, Σ, Ω
        - 수학 기호: ∑, ∏, ∫, ∂, ∇, √, ∞, ≤, ≥, ≈, ≠, ±, ×, ÷
    
    Args:
        text (str): 원본 텍스트
    
    Returns:
        str: 수식이 <formula>로 치환된 텍스트
    
    Examples:
        >>> replace_formulas("minimize $L = \\sum_{i=1}^{n} x_i$")
        'minimize <formula>'
        
        >>> replace_formulas("using α-divergence with β=0.5")
        'using <formula>-divergence with <formula>=<digit>'
    """
    if not isinstance(text, str):
        return text
    
    # 1. LaTeX 형식 치환 (탐욕적이지 않게 처리)
    text = re.sub(r'\$\$[^$]+\$\$', '<formula>', text)  # $$...$$
    text = re.sub(r'\$[^$]+\$', '<formula>', text)      # $...$
    text = re.sub(r'\\\([^)]+\\\)', '<formula>', text)  # \(...\)
    text = re.sub(r'\\\[[^\]]+\\\]', '<formula>', text) # \[...\]
    
    # 2. 그리스 문자 (소문자, 대문자)
    greek_letters = r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]'
    text = re.sub(greek_letters, '<formula>', text)
    
    # 3. 수학 연산자 및 기호
    math_symbols = r'[∑∏∫∂∇√∞≤≥≈≠±×÷⊕⊗⊙∈∉⊂⊃∪∩]'
    text = re.sub(math_symbols, '<formula>', text)
    
    return text


def replace_digits(text):
    """
    텍스트의 모든 숫자를 <digit> 토큰으로 치환
    
    Args:
        text (str): 원본 텍스트
    
    Returns:
        str: 숫자가 <digit>으로 치환된 텍스트
    
    Examples:
        >>> replace_digits("The model achieved 95.3% accuracy in 2024")
        'The model achieved <digit>% accuracy in <digit>'
        
        >>> replace_digits("Use 1000 samples with learning rate 0.001")
        'Use <digit> samples with learning rate <digit>'
    """
    if not isinstance(text, str):
        return text
    
    # 정수 및 소수점 숫자를 모두 <digit>으로 치환
    # \d+\.?\d* : 정수 또는 소수 (예: 123, 45.67, 0.5)
    text = re.sub(r'\d+\.?\d*', '<digit>', text)
    
    return text


def normalize_whitespace(text):
    """
    연속된 공백을 단일 공백으로 정규화하고 앞뒤 공백 제거
    
    Args:
        text (str): 원본 텍스트
    
    Returns:
        str: 공백이 정규화된 텍스트
    
    Examples:
        >>> normalize_whitespace("Deep   learning    for  NLP")
        'Deep learning for NLP'
        
        >>> normalize_whitespace("  This  is   a    test  ")
        'This is a test'
        
        >>> normalize_whitespace("Text with\\n\\nnewlines\\tand\\ttabs")
        'Text with newlines and tabs'
    
    Note:
        - 공백, 탭, 개행 등 모든 whitespace를 단일 공백으로 치환
        - PDF 변환 과정에서 생긴 불규칙한 공백 정리에 유용
    """
    if not isinstance(text, str):
        return text
    
    # 연속된 whitespace(공백, 탭, 개행 등)를 단일 공백으로 치환
    text = re.sub(r'\s+', ' ', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text


def build_columns(df):
    df = df.copy()
    
    # 수식 치환 → 숫자 치환 → 공백 정규화 (순서 중요!)
    df["title"] = (df["title"]
                   .apply(replace_formulas)
                   .apply(replace_digits)
                   .apply(normalize_whitespace))
    df["abstract"] = (df["abstract"]
                      .apply(replace_formulas)
                      .apply(replace_digits)
                      .apply(normalize_whitespace))

    df["present_kps"] = df.apply(
        lambda r: [kp for kp, p in zip(r["keyphrases"], r["prmu"]) if p == "P"],
        axis=1
    )

    df["absent_kps"] = df.apply(
        lambda r: [kp for kp, p in zip(r["keyphrases"], r["prmu"]) if p != "P"],
        axis=1
    )

    df["source_text"] = df.apply(
        lambda r: f"generate keyphrases: title: {r['title']} abstract: {r['abstract']}",
        axis=1
    )

    df["target_text"] = df["keyphrases"].apply(lambda kps: str(np.array(kps)))

    return df

def build_inference_columns(df):
    """
    추론용 컬럼 생성: source_text만 생성
    
    Args:
        df (pd.DataFrame): title, abstract 컬럼이 있는 DataFrame
    
    Returns:
        pd.DataFrame: source_text 컬럼이 추가된 DataFrame
    
    Note:
        - keyphrases가 없어도 동작
        - target_text, present_kps, absent_kps는 생성하지 않음
    """
    df = df.copy()
    
    # 수식 치환 → 숫자 치환 → 공백 정규화 (순서 중요!)
    df["title"] = (df["title"]
                   .apply(replace_formulas)
                   .apply(replace_digits)
                   .apply(normalize_whitespace))
    df["abstract"] = (df["abstract"]
                      .apply(replace_formulas)
                      .apply(replace_digits)
                      .apply(normalize_whitespace))
    
    df["source_text"] = df.apply(
        lambda r: f"generate keyphrases: title: {r['title']} abstract: {r['abstract']}",
        axis=1
    )
    
    return df


def save_to_arrow(df, path):
    os.makedirs(path, exist_ok=True)
    dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
    dataset.save_to_disk(path)
    print(f"저장 완료: {path}")
