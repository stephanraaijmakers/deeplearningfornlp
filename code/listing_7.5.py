
def discretize_attention(normalized_attention):
    multiplier=1
    for p in normalized_attention:
        p_str=str(p)
        if 'e-0' in p_str:
            p_decomposed=p_str.split(".")
            if len(p_decomposed)>1:
                decimals=p_decomposed[1].split("e-0")
                x=10**(len(decimals[0])+int(decimals[1]))
        else:
            x=10**(1+np.floor(np.log10(np.abs(x)))*-1)
        if x>multiplier:
            multiplier=x
    discrete_attention=[]
    for p in normalized_attention:
        discrete_attention.append(int(p*multiplier))
    return discrete_attention

