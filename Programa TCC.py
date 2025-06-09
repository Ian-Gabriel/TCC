    import numpy as np
    from CoolProp.CoolProp import PropsSI
    from scipy.optimize import fsolve
    
    #from scipy.optimize import fsolve
    #import matplotlib.pyplot as plt
    
    # ===========================================================================
    # ETAPA 0: DEFINIÇÃO DOS DADOS DE ENTRADA
    # ===========================================================================
    
    # Dados Gerais
    T_amb_C = 25                                                                     # Teemperatura ambiente [°C]
    T_amb_K = T_amb_C + 273.15                                                                     # Teemperatura ambiente [°C]
    P_amb_Pa = 1.01325*1e5                                                           # Pressão ambiente [bar]
    U_s = 70                                                                         # Coeficientes de Transferência de Calor [kcal/m²h°C]
    U_r = 70                                                                         # Coeficientes de Transferência de Calor [kcal/m²h°C]
    U_v = 80                                                                         # Coeficientes de Transferência de Calor [kcal/m²h°C]
    U_e = 75                                                                         # Coeficientes de Transferência de Calor [kcal/m²h°C]
    U_a = 40                                                                         # Coeficientes de Transferência de Calor [kcal/m²h°C]
    f_F = 0.4                                                                        # Fator de Perda (Fornalha)
    f_S = 0.15
    f_R = f_S
    f_V = 0.1
    f_E = f_V
    f_A = f_V
    beta = 4.875*1e-8
    
    # Cenários
    cenarios = {
        "Cenário 01":{
        "m_v_nom_tph": 100,                                                         # Produção de vapor nominal [ton/h]
        "P_superaq_bar_man": 70,                                                    # Pressão superaquecedor [bar] (manométrica)
        "T_superaq_C": 450,                                                         # Temperatura superaquecedor [°C]
        "P_reaq_bar_man": 8,                                                        # Pressão reaquecedor [bar] (manométrica)
        "T_reaq_C": 440,                                                            # Temperatura reaquecedor [°C]
        "m_e_fracao": 0.90,                                                         # Fração mássica de reaquecimento 
        "theta_1_C": 250,                                                           # Temperatura da água de alimentação (entrada economizador) [°C]
                                                             
        },        
    
        "Cenário 02":{
        "m_v_nom_tph": 120,                                                         # Produção de vapor nominal [ton/h]
        "P_superaq_bar_man": 80,                                                    # Pressão superaquecedor [bar] (manométrica)
        "T_superaq_C": 475,                                                         # Temperatura superaquecedor [°C]
        "P_reaq_bar_man": 10,                                                       # Pressão reaquecedor [bar] (manométrica)
        "T_reaq_C": 465,                                                            # Temperatura reaquecedor [°C]
        "m_E_fracao": 0.85,                                                         # Fração mássica de reaquecimento
        "theta_1_C": 255,                                                           # Temperatura da água de alimentação (entrada economizador) [°C]                                                        
        },  
        
        "Cenário 03":{
        "m_v_nom_tph": 140,                                                         # Produção de vapor nominal [ton/h]
        "P_superaq_bar_man": 90,                                                    # Pressão superaquecedor [bar] (manométrica)
        "T_superaq_C": 485,                                                         # Temperatura superaquecedor [°C]
        "P_reaq_bar_man": 12,                                                       # Pressão reaquecedor [bar] (manométrica)
        "T_reaq_C": 475,                                                            # Temperatura reaquecedor [°C]
        "m_E_fracao": 0.80,                                                         # Fração mássica de reaquecimento
        "theta_1_C": 260,                                                           # Temperatura da água de alimentação (entrada economizador) [°C]  
        },  
        
        "Cenário 04":{
        "m_v_nom_tph": 160,                                                         # Produção de vapor nominal [ton/h]
        "P_superaq_bar_man": 100,                                                   # Pressão superaquecedor [bar] (manométrica)
        "T_superaq_C": 500,                                                         # Temperatura superaquecedor [°C]
        "P_reaq_bar_man": 14,                                                       # Pressão reaquecedor [bar] (manométrica)
        "T_reaq_C": 490,                                                            # Temperatura reaquecedor [°C]
        "m_E_fracao": 0.75, 
        "theta_1_C": 265,                                                           # Temperatura da água de alimentação (entrada economizador) [°C]                                                        # Fração mássica de reaquecimento
        }  
    }     
            
    #Combustível
    combustiveis = {
        "Bagaço de Cana":{
        "tipo_combustivel": "Bagaço de Cana",                                       # Tipo de combustível
        "porcentagem_cinzas": 5,                                                    # Teor de cinzas [%]
        "PCI_kcal_kg": 2100,                                                        # Poder Calorífico Inferior [kcal/kg]
        "P1_perda_comb_cinzas_perc": 2,                                             # Perda por combustíveis nas cinzas [% do combustível]
        "P2_perda_calor_sens_cinzas_perc": 0,                                       # Perda pelo calor sensível contido nas cinzas    
        "P3_perda_form_fuligem_perc": 0,                                            # Perda por formação de fuligem  
        "P4_perda_incomp_comb_perc": 4,                                             # Perda por combustão incompleta [% do combustível]
        "t_cb_C": T_amb_C,                                                            # Temperatura do combustível [°C]
        "emissividade_fornalha": 0.55,                                              # Emissividade da fornalha (beta * epsilon)
        "AC0":lambda PCI_kcal_kg: 1.3*10**-3 * PCI_kcal_kg + 0.32,                  # Relação A/C estequiométrica [kg ar / kg cb]
        "lambda_min": 1.30,                                                         # 30% de excesso de ar
        "lambda_max": 1.60,                                                         # 60% de excesso de ar
        "grupo_cpg": "oleo_carvao_bagaco"
        },
        
        "Carvão Pulverizado":{
        "tipo_combustivel": "Carvão Pulverizado",                                   # Tipo de combustível
        "porcentagem_cinzas": 25,                                                   # Teor de cinzas [%]
        "PCI_kcal_kg": 5000,                                                        # Poder Calorífico Inferior [kcal/kg]
        "P1_perda_comb_cinzas_perc": 4,                                             # Perda por combustíveis nas cinzas [% do combustível]
        "P2_perda_calor_sens_cinzas_perc": 0,                                       # Perda pelo calor sensível contido nas cinzas    
        "P3_perda_form_fuligem_perc": 0,                                            # Perda por formação de fuligem  
        "P4_perda_incomp_comb_perc": 3,                                             # Perda por combustão incompleta [% do combustível]
        "t_cb_C": T_amb_C,                                                            # Temperatura do combustível [°C]
        "emissividade_fornalha": 0.80,                                              # Emissividade da fornalha (beta * epsilon)
        "AC0":lambda PCI_kcal_kg: 1.38*10**-3 * PCI_kcal_kg,                        # Relação A/C estequiométrica [kg ar / kg cb]
        "lambda_min": 1.15,                                                         # 15% de excesso de ar
        "lambda_max": 1.25,                                                         # 25% de excesso de ar
        "grupo_cpg": "oleo_carvao_bagaco"
        },
        
        "Óleo Combustível":{
        "tipo_combustivel": "Óleo Combustível",                                     # Tipo de combustível
        "porcentagem_cinzas": 0.5,                                                  # Teor de cinzas [%]
        "PCI_kcal_kg": 9600,                                                        # Poder Calorífico Inferior [kcal/kg]
        "P1_perda_comb_cinzas_perc": 0.2,                                           # Perda por combustíveis nas cinzas [% do combustível]
        "P2_perda_calor_sens_cinzas_perc": 0,                                       # Perda pelo calor sensível contido nas cinzas    
        "P3_perda_form_fuligem_perc": 0,                                            # Perda por formação de fuligem  
        "P4_perda_incomp_comb_perc": 2,                                             # Perda por combustão incompleta [% do combustível]
        "t_cb_C": 150,                                                              # Temperatura do combustível [°C]
        "emissividade_fornalha": 0.75,                                              # Emissividade da fornalha (beta * epsilon)
        "AC0":lambda PCI_kcal_kg: 1.38*10**-3 * PCI_kcal_kg,                        # Relação A/C estequiométrica [kg ar / kg cb]
        "lambda_min": 1.15,                                                         # 15% de excesso de ar
        "lambda_max": 1.25,                                                         # 25% de excesso de ar
        "grupo_cpg": "oleo_carvao_bagaco"
        },  
    
        "Gás Natural":{
        "tipo_combustivel": "Gás Natural",                                          # Tipo de combustível
        "porcentagem_cinzas": 0,                                                    # Teor de cinzas [%]
        "PCI_kcal_kg": 10500,                                                       # Poder Calorífico Inferior [kcal/kg]
        "P1_perda_comb_cinzas_perc": 0,                                             # Perda por combustíveis nas cinzas [% do combustível]
        "P2_perda_calor_sens_cinzas_perc": 0,                                       # Perda pelo calor sensível contido nas cinzas    
        "P3_perda_form_fuligem_perc": 0,                                            # Perda por formação de fuligem  
        "P4_perda_incomp_comb_perc": 1,                                             # Perda por combustão incompleta [% do combustível]
        "t_cb_C": T_amb_C,                                                          # Temperatura do combustível [°C]
        "emissividade_fornalha": 0.30,                                              # Emissividade da fornalha (beta * epsilon)
        "AC0":lambda PCI_kcal_kg: 1.5*10**-3 * PCI_kcal_kg,                         # Relação A/C estequiométrica [kg ar / kg cb]
        "lambda_min": 1.05,                                                         # 5% de excesso de ar
        "lambda_max": 1.10,                                                         # 10% de excesso de ar
        "grupo_cpg": "gn"
        }   
    }
    
    # Dados de calor específico dos gases (Cpg) por grupo de combustível
    dados_c_pg_grupos = {
        "oleo_carvao_bagaco": {
            "cp_kcal_kg0C": 0.25 ,
            "cp_kcal_kg1200C": 0.30
        },
        "gn": {
            "cp_kcal_kg0C": 0.26 ,
            "cp_kcal_kg1200C": 0.32                                                 # Valores do PDF para GN
        }
    }
    
    
    # ===========================================================================
    # ETAPA 1: ESCOLHER PARÂMETROS E COMBUSTÍVEL
    # ===========================================================================
    cenario_escolhido = "Cenário 01"
    combustivel_escolhido = "Bagaço de Cana"
    
    cenario = cenarios[cenario_escolhido]
    combustivel = combustiveis[combustivel_escolhido]
    
    # ===========================================================================
    # ETAPA 2: DEFINIÇÃO DAS NOVAS VARIÁVEIS
    # ===========================================================================
    
    # Extração de dados do cenário
    m_v           = cenario["m_v_nom_tph"]*1000                                     # Produção de vapor nominal [kg/h]
    m_e           = (cenario["m_e_fracao"])                                         # Fração mássica de reaquecimento 
    P_SR_Pa       = cenario["P_superaq_bar_man"]* 1e5 + P_amb_Pa                              # Pressão do superaquecedor [Pa]
    T_SR_C        = cenario["T_superaq_C"]                                          # Temperatura do superaquecedor [°C]
    T_g_C         = 150                                                             # Temperatura de saída dos gases da caldeira (140°C - 180°C)
    T_c_C         = 900                                                             # Temperatura média da fornalha - de acordo com PCI (900°C - 1300°C)
    T_ar_aq_C     = 180                                                             # Temperatura de aquecimento do ar - de acordo com o combustível (150°C - 300 °C)
    P_RV_Pa       = cenario["P_reaq_bar_man"]* 1e5 + P_amb_Pa                                  # Pressão do reaquecedor [Pa]
    T_RV_C        = cenario["T_reaq_C"]                                             # Temperatura do reaquecedor [°C]
    theta1_C      = cenario["theta_1_C"]                                            # Temperatura da água na entrada do economizador [°C]
    #melhorar
    
    # Extração de dados do combustível
    PCI           = combustivel["PCI_kcal_kg"]                                      # Poder Calorífico Inferior [kcal/kg]
    z             = combustivel["porcentagem_cinzas"]/100                           # Teor de cinzas [%]
    P1            = combustivel["P1_perda_comb_cinzas_perc"] / 100                  # Perda por cinzas
    P2            = combustivel["P2_perda_calor_sens_cinzas_perc"] / 100            # Perda pelo calor sensível contido nas cinzas
    P3            = combustivel["P3_perda_form_fuligem_perc"] / 100                 # Perda por formação de fuligem
    P4            = combustivel["P4_perda_incomp_comb_perc"] / 100                  # Perda por combustão incompleta
    T_cb_C        = combustivel["t_cb_C"]                                           # Temperatura do combustível [°C]
    emis_C        = combustivel["emissividade_fornalha"]                            # Emissividade da fornalha
    lambda_min    = combustivel["lambda_min"]                                       # Excesso mínimo de ar
    lambda_max    = combustivel["lambda_max"]                                       # Excesso máximo de ar
    grupo_cpg     = combustivel["grupo_cpg"]
    Cp_0C         = dados_c_pg_grupos[grupo_cpg]["cp_kcal_kg0C"]                    # Grupo para Cp dos gases 0°C
    Cp_1200C      = dados_c_pg_grupos[grupo_cpg]["cp_kcal_kg1200C"]                 # Grupo para Cp dos gases 1200°C
    C_cb         = 0                                                                # Calor específico do Combustível
    AC0           = combustivel["AC0"](PCI)
    
    
    # Conversão de Unidades                                                      
    T_SR_K = T_SR_C + 273.15                                                     
    T_RV_K = T_RV_C + 273.15
    theta1_K = theta1_C + 273.15
    T_g_K    = T_g_C + 273.15                                                    # Temperatura de saída dos gases da caldeira (140°C - 180°C)
    T_c_K    = T_c_C + 273.15                                                    # Temperatura média da fornalha - de acordo com PCI (900°C - 1300°C)
    T_ar_aq_K = T_ar_aq_C + 273.15                                               # Temperatura de aquecimento do ar - de acordo com o combustível (150°C - 300 °C)
    
    # ===========================================================================
    # ETAPA 3: CÁLCULOS PRELIMINARES
    # ===========================================================================
    print("CÁLCULOS PRELIMINARES:")
    
    # Relação ar/combustível (AC)
    AC = ((lambda_min + lambda_max)/2) * AC0
    print(f"Relação ar/combustível (AC): {AC:.4f}")
    
    # Temperatura média dos gases na chaminé (T_g_c_C)
    T_g_c_C = (T_g_C + T_amb_C)/2.0                                                 # Celsius
    print(f"Temperatura média dos gases na chaminé (T_g_c_C): {T_g_c_C:.4f} °C")
    
    # Interpolação para a temperatura média dos gases da chaminé
    Cp_g_c = Cp_0C + (Cp_1200C - Cp_0C) * (T_g_c_C / 1200)
    print(f"Capacidade Térmica dos Gases na Fornalha (Cp_g_c): {Cp_g_c:.4f} °kcal/kg")
    
    # Perda pelo calor contido nos gases da chaminé (P6)
    P6 = ((AC+1-z)*Cp_g_c*(T_g_C - T_amb_C))/PCI
    print(f"Calor contido nos gases da chaminé (P6): {P6:.4f}")
    
    # Rendimento da fornalha (eta_F)
    eta_F = 1 - (P1 + P2 + P3 + P4)
    print(f"Rendimento da Fornalha (eta_F): {eta_F:.4f}")
    
    # Perda do calor para o ambiente (P5)
    P5 = 1.2/100                                                                    # Retirado do Gráfico
    print(f"Perda do calor para o ambiente (P5): {P5:.4f}")
    
    # Rendimento da caldeira (eta)
    eta = 1 - (P1 + P2 + P3 + P4 + P5 + P6)
    print(f"Rendimento da caldeira (eta): {eta:.4f}")
    
    # Cálculo entalpias
    # h_s: Saída do superaquecedor - vapor superaquecido (h_s)
    h_s = PropsSI("H", "P", P_SR_Pa, "T", T_SR_K, "Water") / 4186.8                # kcal/kg
    print(f"Saída do superaquecedor - vapor superaquecido (h_s): {h_s:.4f}")
    # h_1: Entrada da água de alimentação (h_1)
    h_1 = PropsSI("H", "P", P_SR_Pa, "T", theta1_K, "Water") / 4186.8               # kcal/kg
    print(f"Entrada da água de alimentação (h_1): {h_1:.4f}")
    # h_e: Entrada do reaquecedor - vapor saturado seco (h_e)
    h_e = PropsSI("H", "P", P_RV_Pa, "Q", 1, "Water") / 4186.8                      # kcal/kg
    print(f"Entrada do reaquecedor - vapor saturado seco (h_e): {h_e:.4f}")
    # h_r: Saída do reaquecedor - vapor superaquecido (h_r)
    h_r = PropsSI("H", "P", P_RV_Pa, "T", T_RV_K, "Water") / 4186.8                 # kcal/kg
    print(f"Saída do reaquecedor - vapor superaquecido (h_r): {h_r:.4f}")
    
    # Consumo de combustível (m_dot_cb)
    m_dot_cb = (m_v * (h_s - h_1) + (m_e *m_v)* (h_r - h_e))\
                / (eta * PCI + C_cb * (T_cb_C - T_amb_C))                               # kg/h
    print(f"Consumo de combustível (m_dot_cb): {m_dot_cb:.4f} kg/h")
    
    # Potência Térmica Fornecida na fornalha (Q5)
    Q_5 = P5 * m_dot_cb * PCI                                                       # kcal/h
    print(f"Potência Térmica Fornecida na fornalha (Q5): {Q_5:.4f} kcal/h")
    
    print("FIM DA ETAPA 3")
    print()
    
    # ===========================================================================
    # ETAPA 4: CÁLCULOS FORNALHA
    # ===========================================================================
    print("ETAPA 4: CÁLCULOS FORNALHA")
    
    # Temperatura média dos gases na fornalha (T_g_F_C)
    T_g_F_C = (T_c_C + T_amb_C)/2.0                                                 # Celsius
    print(f"Temperatura média dos gases na fornalha (T_g_F_K): {T_g_F_C:.4f} °C")
    
    # Interpolação para a temperatura média da Fornalha (Cp_g_F):
    Cp_g_F = Cp_0C + (Cp_1200C - Cp_0C) * (T_g_F_C / 1200)
    print(f"Capacidade Térmica dos Gases na Fornalha (Cp_g_F): {Cp_g_F:.4f} °kcal/kg")
    
    # Temperatura média do ar na fornalha (T_ar_F)
    T_ar_F = (T_ar_aq_K + T_amb_K)/2.0 
    print(f"Temperatura média do ar na fornalha (T_ar_F): {(T_ar_F):.4f} K")
    
    # Capacidade térmica do ar na Fornalha (Cp_ar_F)
    Cp_ar_F = PropsSI('C', 'T', T_ar_F, 'P', P_amb_Pa, 'Air')/4186.8
    print(f"Capacidade térmica do ar na Fornalha (Cp_ar_F): {(Cp_ar_F):.4f} kcal/kg")
    
    # Definição da função para resolver sigma
    def equacao_sigma(sigma):
        numerador = (1 - sigma) * eta_F * PCI + \
                    AC * Cp_ar_F * (T_ar_aq_K - T_amb_K) + \
                    C_cb * (T_cb_C - T_amb_C) - \
                    f_F * P5 * PCI
        denominador = (AC + 1 - z) * Cp_g_F
        return T_c_C - ((numerador / denominador) + T_amb_C)                        # Celsius
    
    # Resolvendo o Coeficiente de irradiação da fornalha (σ)
    sigma = fsolve(equacao_sigma, 0.1)[0]
    print(f"Coeficiente de irradiação da fornalha (σ): {sigma:.6f}")
      
    # Calor irradiado na fornalha para as paredes de água da fornalha (Qi)
    Q_i = sigma * (m_dot_cb * eta_F * PCI)                                               #kcal
    print(f"Calor irradiado na fornalha para as paredes de água (Q_i): {Q_i:.6f}")
    
    # Temperatura do vapor saturado (T_v)
    T_v_K = PropsSI("T", "P", P_SR_Pa, "Q", 0, "Water")                                # Q=0 para líquido saturado
    T_v_C = T_v_K - 273.15
    print(f"Temperatura do vapor saturado (T_v_K): {T_v_K:.4f} K")
    
    # Temperatura dos tubos de parede d'água (T_p_K)
    T_p_K = T_v_K + 30                                                                   # Usar 20°C - 40°C
    print(f"Temperatura dos tubos de parede d'água (T_p_K): {T_p_K:.4f} K") 
     
    # Superfície irradiada normal a chama - CONFERIR VALOR DE Fp                                                                                                                                      
    S_i = Q_i / (beta * emis_C * ((T_c_K)**4 - T_p_K**4))                               # m² 
    print(f"Superfície irradiada normal a chama (S_i): {S_i:.6f} m²")
    
    print("FIM DA ETAPA 4")
    print()
    
    # ===========================================================================
    # ETAPA 5: CÁLCULOS SUPERAQUECEDOR
    # ===========================================================================
    print("ETAPA 5: CÁLCULOS SUPERAQUECEDOR")
    
    # Entrada do superaquecedor - vapor saturado (h_v)
    h_v = PropsSI("H", "P", P_SR_Pa, "Q", 1, "Water")/ 4186.8                         # Usar "Q" (qualidade) = 1 para vapor saturado seco
    print(f"Entrada do superaquecedor - vapor saturado (h_v): {h_v:.4f} kcal/kg")
    
    # Calor requerido para o superaquecimento de vapor (Q_s)
    Q_s = m_v*(h_s - h_v) 
    print(f"Calor requerido para o superaquecimento de vapor (Q_s): {Q_s:.4f} kcal")
    
    # Iteração para encontrar temperatura média no superaquecedor (T_sr):
    T_c_C_N = T_c_C                                                                       # Tc (saída da fornalha) já está em °C (ex: 900°C)
    T_SR_C_estimada = T_c_C_N - 100                                                     # Uma estimativa inicial para Tsr em °C
    Cp_g_SR_atual = 0.0                                                             # Inicializar
    
    for _iteracao in range(10):                                                     # Loop simples de 10 iterações (pode precisar de mais ou menos)
        T_media_gases_SR_C = (T_c_C_N + T_SR_C_estimada) / 2.0
        temp_interp = np.clip(T_media_gases_SR_C, 0, 1200)                         # Limite para a interpolação
        Cp_g_SR_atual = Cp_0C + (Cp_1200C - Cp_0C) * (temp_interp / 1200.0)
    
        denominador_T_SR = m_dot_cb * (AC + 1 - z) * Cp_g_SR_atual
        if denominador_T_SR == 0:
            print("ERRO: Denominador zero ao calcular Tsr!")
            break
        
        T_SR_C_nova = T_c_C_N - (Q_s + f_S * Q_5) / denominador_T_SR
        
        if abs(T_SR_C_nova - T_SR_C_estimada) < 0.1: # Critério de convergência
            T_SR_C_estimada = T_SR_C_nova
            break
        T_SR_C_estimada = T_SR_C_nova
    else:
        print("ALERTA: T_SR_C não convergiu satisfatoriamente.")
    
    T_g_SR_C = T_SR_C_estimada
    print(f"Temperatura dos gases na saída do superaquecedor (T_g_SR_C): {T_g_SR_C:.4f} °C")
    
    # Diferença Logarítimixa de Temperaturas
    Delta_T1_SR = (T_c_C - T_SR_C)
    print(f"Delta_T1_SR: {Delta_T1_SR:.4f} °C")
    Delta_T2_SR = (T_g_SR_C - T_v_C)
    print(f"Delta_T2_SR: {Delta_T2_SR:.4f} °C")
    Delta_T_LS_SR = (Delta_T1_SR - Delta_T2_SR)/np.log(Delta_T1_SR/Delta_T2_SR)
    print(f"Diferença Logarítimixa de Temperaturas (Delta_T_LS_SR): {Delta_T_LS_SR:.4f} °C")
    
    # Cálculo da área superficial do superaquecedor (S_s)
    S_S = Q_s/(U_s * Delta_T_LS_SR)                                                   # m²
    print(f"Cálculo da área superficial do superaquecedor (S_s): {S_S:.6f} m²")
    
    print("FIM DA ETAPA 5")
    print()
    
    # ===========================================================================
    # ETAPA 6: CÁLCULO DO REAQUECEDOR
    # ===========================================================================
    print("ETAPA 6: CÁLCULO DO REAQUECEDOR")
    
    # Temperatura e entalpia na entrada do reaquecedor
    T_e_K = PropsSI("T", "P", P_RV_Pa, "Q", 1, "Water")                             # Temp. saturação em Kelvin
    print(f"Entrada do reaquecedor - vapor saturado (T_e_K): {T_e_K:.4f} K")
    T_e_C = T_e_K - 273.15  
    print(f"Entrada do reaquecedor - vapor saturado (T_e_C): {T_e_C:.4f} K")
    h_e = PropsSI("H", "P", P_RV_Pa, "Q", 1, "Water") / 4186.8                      # kcal/kg
    print(f"Entrada do reaquecedor - vapor saturado (h_e): {h_e:.4f} kcal/kg")
    
    # Calor requerido para o reaquecimento do vapor (Q_r)
    Q_r = m_e * m_v * (h_r - h_e)
    print(f"Calor requerido para o reaquecimento do vapor (Q_r): {Q_r:.4f} kcal/kg")
    
    # Iteração para encontrar temperatura média no superaquecedor (T_RV):
    T_SR_C_N = T_g_SR_C                                                                       # Tc (saída da fornalha) já está em °C (ex: 900°C)
    T_RV_C_estimada = T_SR_C_N - 250                                                     # Uma estimativa inicial para Tsr em °C
    Cp_g_RV_atual = 0.0                                                             # Inicializar
    
    for _iteracao in range(10):                                                     # Loop simples de 10 iterações (pode precisar de mais ou menos)
        T_media_gases_RV_C = (T_SR_C_N + T_RV_C_estimada) / 2.0
        temp_interp = np.clip(T_media_gases_RV_C, 0, 1200)                         # Limite para a interpolação
        Cp_g_RV_atual = Cp_0C + (Cp_1200C - Cp_0C) * (temp_interp / 1200.0)
    
        denominador_T_RV = m_dot_cb * (AC + 1 - z) * Cp_g_RV_atual
        if denominador_T_RV == 0:
            print("ERRO: Denominador zero ao calcular T_RV!")
            break
        
        T_RV_C_nova = T_SR_C_N - (Q_r + f_R * Q_5) / denominador_T_RV
        
        if abs(T_RV_C_nova - T_RV_C_estimada) < 0.1: # Critério de convergência
            T_RV_C_estimada = T_RV_C_nova
            break
        T_RV_C_estimada = T_RV_C_nova
    else:
        print("ALERTA: T_SR_C não convergiu satisfatoriamente.")
    
    T_g_RV_C = T_RV_C_estimada
    print(f"Temperatura dos gases na saída do superaquecedor (T_g_RV_C): {T_g_RV_C:.4f} °C")
    
    # Diferença Logarítimica de Temperaturas
    Delta_T1_RV = (T_g_SR_C - T_RV_C)
    print(f"Delta_T1_RV: {Delta_T1_RV:.4f} °C")
    
    Delta_T2_RV = (T_g_RV_C - T_e_C)
    print(f"Delta_T2_RV: {Delta_T2_RV:.4f} °C")
    
    Delta_T_LS_RV = (Delta_T1_RV - Delta_T2_RV) / np.log(Delta_T1_RV / Delta_T2_RV)
    print(f"Diferença Logarítimixa de Temperaturas (Delta_T_LS_RV): {Delta_T_LS_RV:.4f} °C")
    
    # Cálculo da área superficial do superaquecedor (S_s):
    S_R = Q_r/  (U_r * Delta_T_LS_RV)                                                   # m²
    print(f"Cálculo da área superficial do reaquecedor (S_R): {S_R:.6f} m²")
    
    print("FIM DA ETAPA 6")
    print()
     
    # ===========================================================================
    # ETAPA 7: CÁLCULO DO VAPORIZADOR (PARTE CONVECTIVA)
    # ===========================================================================
    print("ETAPA 7: CÁLCULO DO VAPORIZADOR (PARTE CONVECTIVA)")
    
    # Temperatura e entalpia na entrada do vaporizador
    # Temperatura - theta2 (recomendação)
    theta2_K = T_v_K - 30
    theta2_C = theta2_K - 273.15
    print(f"Entrada no vaporizador - vapor saturado (theta2_K): {theta2_K}K")
    h_2 = PropsSI("H", "P", P_SR_Pa, "T", theta2_K, "Water") / 4186.8                      # kcal/kg # J/kg
    print(f"Entrada no vaporizador - vapor saturado (h_2): {h_2}kcal/kg")
    
    # Calor requerido para a vaporização da água (Q_v)
    Q_v = m_v * (h_v - h_2) - Q_i
    print(f"Calor requerido para a vaporização da água (Q_v): {Q_v} kcal/h")
    
    # Iteração para encontrar temperatura média no superaquecedor (T_RV):
    T_RV_C_N = T_g_RV_C                                                                       # Tc (saída da fornalha) já está em °C (ex: 900°C)
    T_VE_C_estimada = T_RV_C_N - 250                                                     # Uma estimativa inicial para Tsr em °C
    Cp_g_VE_atual = 0.0                                                             # Inicializar
    
    for _iteracao in range(10):                                                     # Loop simples de 10 iterações (pode precisar de mais ou menos)
        T_media_gases_VE_C = (T_RV_C_N + T_VE_C_estimada) / 2.0
        temp_interp = np.clip(T_media_gases_VE_C, 0, 1200)                         # Limite para a interpolação
        Cp_g_VE_atual = Cp_0C + (Cp_1200C - Cp_0C) * (temp_interp / 1200.0)
    
        denominador_T_VE = m_dot_cb * (AC + 1 - z) * Cp_g_VE_atual
        if denominador_T_VE== 0:
            print("ERRO: Denominador zero ao calcular T_VE!")
            break
        
        T_VE_C_nova = T_RV_C_N - (Q_v + f_V * Q_5) / denominador_T_VE    
        if abs(T_VE_C_nova - T_VE_C_estimada) < 0.1: # Critério de convergência
            T_VE_C_estimada = T_VE_C_nova
            break
        T_VE_C_estimada = T_VE_C_nova
    else:
        print("ALERTA: T_VE_C não convergiu satisfatoriamente.")
    
    T_g_VE_C = T_VE_C_estimada
    print(f"Temperatura dos gases na saída do vaporizador (T_g_VE_C): {T_g_VE_C:.4f} °C")
    
    # Diferença Logarítimica de Temperaturas
    Delta_T1_VE = (T_g_RV_C - T_v_C)
    print(f"Delta_T1_VE: {Delta_T1_VE:.4f} °C")
    
    Delta_T2_VE = (T_g_VE_C - T_v_C)
    print(f"Delta_T2_VE {Delta_T2_VE:.4f} °C")
    
    Delta_T_LS_VE = (Delta_T1_VE - Delta_T2_VE) / np.log(Delta_T1_VE / Delta_T2_VE)
    print(f"Diferença Logarítimica de Temperaturas (Delta_T_LS_VE): {Delta_T_LS_VE:.4f} °C")
    
    # Cálculo da área superficial do vaporizador (S_V):
    S_V = Q_v/  (U_v * Delta_T_LS_VE)                                                   # m²
    print(f"Cálculo da área superficial do vaporizador (S_V): {S_V:.6f} m²")
    
    print("FIM DA ETAPA 7")
    print()
    
    # ===========================================================================
    # ETAPA 8: CÁLCULO DO ECONOMIZADOR
    # ===========================================================================
    print("ETAPA 8: CÁLCULO DO ECONOMIZADOR")
    
    # Calor requerido para o aquecimento da água (Q_e)
    Q_e = m_v * (h_2 - h_1)
    print(f"Calor requerido para o aquecimento da água (Q_e): {Q_e} kcal/h")
    
    # Iteração para encontrar temperatura média no economizador (T_EA):
    T_VE_C_N = T_g_VE_C                                                                       # Tc (saída da fornalha) já está em °C (ex: 900°C)
    T_AE_C_estimada = T_VE_C_N - 100                                                     # Uma estimativa inicial para Tsr em °C
    Cp_g_VE_atual = 0.0    
    
    # Iteração para encontrar temperatura média no superaquecedor (T_RV):
    T_VE_C_N = T_g_VE_C                                                                       # Tc (saída da fornalha) já está em °C (ex: 900°C)
    T_AE_C_estimada = T_VE_C_N - 100                                                    # Uma estimativa inicial para Tsr em °C
    Cp_g_AE_atual = 0.0                                                          # Inicializar
    
    for _iteracao in range(10):                                                     # Loop simples de 10 iterações (pode precisar de mais ou menos)
        T_media_gases_AE_C = (T_VE_C_N + T_AE_C_estimada) / 2.0
        temp_interp = np.clip(T_media_gases_AE_C, 0, 1200)                         # Limite para a interpolação
        Cp_g_AE_atual = Cp_0C + (Cp_1200C - Cp_0C) * (temp_interp / 1200.0)
    
        denominador_T_AE = m_dot_cb * (AC + 1 - z) * Cp_g_AE_atual
        if denominador_T_AE== 0:
            print("ERRO: Denominador zero ao calcular T_VE!")
            break
        
        T_AE_C_nova = T_VE_C_N - (Q_e + f_E * Q_5) / denominador_T_AE
        
        if abs(T_AE_C_nova - T_AE_C_estimada) < 0.1: # Critério de convergência
            T_AE_C_estimada = T_AE_C_nova
            break
        T_AE_C_estimada = T_AE_C_nova
    else:
        print("ALERTA: T_AE_C não convergiu satisfatoriamente.")
    
    T_g_AE_C = T_AE_C_estimada
    print(f"Temperatura média dos gases na saída do economizador (T_g_AE_C): {T_g_AE_C:.4f} °C")
    
    # Diferença Logarítimica de Temperaturas
    Delta_T1_AE = (T_g_VE_C - theta2_C)
    print(f"Delta_T1_AE: {Delta_T1_AE:.4f} °C")
    
    Delta_T2_AE = (T_g_AE_C - theta1_C)
    print(f"Delta_T2_AE {Delta_T2_AE:.4f} °C")
    
    Delta_T_LS_AE = (Delta_T1_AE - Delta_T2_AE) / np.log(Delta_T1_AE / Delta_T2_AE)
    print(f"Diferença Logarítimica de Temperaturas (Delta_T_LS_AE): {Delta_T_LS_AE:.4f} °C")
    
    # Cálculo da área superficial do superaquecedor (S_s):
    S_E = Q_e/  (U_e * Delta_T_LS_AE)                                                   # m²
    print(f"Cálculo da área superficial do reaquecedor (S_E): {S_E:.6f} m²")
    
    print("FIM DA ETAPA 8")
    print()
    
    # ===========================================================================
    # ETAPA 9: CÁLCULO DO AQUECEDOR DE AR
    # ===========================================================================
    
    print("ETAPA 9: CÁLCULO DO AQUECEDOR DE AR")
    
    # Calor requerido para o aquecimento da água (Q_a)
    Q_a = m_dot_cb * AC* Cp_ar_F * (T_ar_aq_C - T_amb_C) + f_A * Q_5 
    print(f"Calor requerido para o aquecimento da ar (Q_a): {Q_a} kcal/h")
    
    # uras
    Delta_T1_AQ = (T_g_AE_C - T_ar_aq_C)
    print(f"Delta_T1_AQ: {Delta_T1_AQ:.4f} °C")
    
    Delta_T2_AQ = (T_g_C - T_amb_C)
    print(f"Delta_T2_AQ: {Delta_T2_AQ:.4f} °C")
    
    Delta_T_LS_AQ = (Delta_T1_AQ - Delta_T2_AQ) / np.log(Delta_T1_AQ / Delta_T2_AQ)
    print(f"Diferença Logarítimica de Temperaturas (Delta_T_LS_AQ): {Delta_T_LS_AQ:.4f} °C")
    
    # Cálculo da área superficial do aquecedor de ar (S_a):
    S_A = Q_a/  (U_a * Delta_T_LS_AQ)                                                   # m²
    print(f"Cálculo da área superficial do reaquecedor (S_A): {S_A:.6f} m²")
    
    # Verificação de Tg
    
    T_g = -(Q_a/(m_dot_cb *(AC + 1 - z) * Cp_g_c)) + T_g_AE_C
    print(f"Cálculo da temperatura dos gasés da chaminé (T_g): {T_g:.6f} °C")
    
    print("FIM DA ETAPA 9")
    print()
    




