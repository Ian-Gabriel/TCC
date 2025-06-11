    import numpy as np
    from CoolProp.CoolProp import PropsSI
    from scipy.optimize import fsolve
    import pandas as pd # <-- ADICIONAR ESTA LINHA
    
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
    T_p_C = T_p_K - 273.15
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

    # ===================================================================================
    # FASE 2: PREPARAÇÃO PARA A SIMULAÇÃO
    # ===================================================================================
    print("\n" + "="*60)
    print("FASE 2: PREPARANDO CONSTANTES PARA A SIMULAÇÃO")
    print("="*60)
    
    # --- Armazenar os Resultados do Projeto Nominal em variáveis "_nom" ---
    # Isso protege os valores originais e deixa o código mais claro.
    m_dot_cb_nom = m_dot_cb
    m_v_nom = m_v
    p5_nom = P5
    U_s_nom, U_r_nom, U_v_nom, U_e_nom, U_a_nom = U_s, U_r, U_v, U_e, U_a
    S_i_nom, S_S_nom, S_R_nom, S_V_nom, S_E_nom, S_A_nom = S_i, S_S, S_R, S_V, S_E, S_A
    h_s_nom, h_v_nom, h_r_nom, h_e_nom, h_1_nom, h_2_nom = h_s, h_v, h_r, h_e, h_1, h_2
    T_SR_C_nom, T_v_C_nom, T_RV_C_nom, T_e_C_nom, theta2_C_nom = T_SR_C, T_v_C, T_RV_C, T_e_C, theta2_C
    
    # --- Calcular Calores Específicos Médios (Constantes para a simulação) ---
    cps = (h_s_nom - h_v_nom) / (T_SR_C_nom - T_v_C_nom) if (T_SR_C_nom - T_v_C_nom) != 0 else 0
    cpr = (h_r_nom - h_e_nom) / (T_RV_C_nom - T_e_C_nom) if (T_RV_C_nom - T_e_C_nom) != 0 else 0
    c = (h_2_nom - h_1_nom) / (theta2_C_nom - theta1_C) if (theta2_C_nom - theta1_C) != 0 else 0
    print(f"Calores específicos calculados: cps={cps:.4f}, cpr={cpr:.4f}, c={c:.4f}")
    
    # ===================================================================================
    # FASE 3: DEFINIÇÃO DAS FUNÇÕES DE CÁLCULO PARA A SIMULAÇÃO
    # ===================================================================================
    #
    # Aqui vamos inserir, uma a uma, as definições das funções para cada componente.
    # Por enquanto, elas são apenas "placeholders".
    #

    def calcular_fornalha_simulacao(
        # --- Parâmetros fixos (constantes da simulação, vindos do cálculo nominal) ---
        S_i,                  # Área da fornalha calculada no projeto nominal [m²]
        T_p_K,                # Temperatura da parede d'água [K] (calculada no nominal)
        eta_F,                # Rendimento da fornalha (ex: 1 - (P1+P2+P3+P4))
        AC,                   # Relação ar/combustível (calculado no nominal)
        
        # --- Parâmetros do combustível (do dicionário 'combustivel') ---
        PCI,                  # PCI do combustível [kcal/kg]
        emis_C,            # Emissividade da fornalha
        cinzas_fracao,        # Teor de cinzas como fração (ex: 5% -> 0.05)
        t_comb,               # Temperatura do combustível [°C]
        C_cb,                 # Calor específico do combustível [kcal/kg°C]
    
        # --- Parâmetros de referência (geralmente fixos) ---
        f_F,                  # Fator de perda da fornalha (ex: 0.4)
        T_amb_C,                # Temperatura ambiente [°C]
        Cp_ar_F,              # Calor específico do ar na fornalha [kcal/kg°C]
        Cp_0C, Cp_1200C,      # Pontos de referência para o Cpg
        
        # --- Parâmetros variáveis (do regime/iteração atual) ---
        m_dot_cb_atual,       # Consumo de combustível do regime atual [kg/h]
        p5_atual,             # Perda p5 corrigida para o regime atual
        t_aq_estimado         # Temperatura do ar aquecido (estimada na iteração) [°C]
        ):
        
        """
        Passo 1 da Simulação: Calcula tc, Qi e sigma para a Fornalha.
        Resolve a dependência circular entre tc e Cpg.
        """
        
        beta = 4.875e-8 # Constante de Stefan-Boltzmann [kcal/m²hK⁴]
        tc_estimado_loop = 1000.0 # Chute inicial para a temperatura da fornalha
        
        # Loop para convergir tc e Cpg
        for i in range(10):
            # Calcula Cpg com base no tc da iteração atual
            T_media_gases = (tc_estimado_loop + T_amb_C) / 2.0
            Cp_g_atual = Cp_0C + (Cp_1200C - Cp_0C) * (T_media_gases / 1200)
    
            # Define a equação para o fsolve resolver
            def equacao_fornalha(tc_array):
                tc = tc_array[0]
                Q_F = eta_F * m_dot_cb_atual * PCI
                if Q_F <= 0: return tc
    
                # Tradução literal da fórmula da fornalha
                termo_Qi = beta * emis_C * S_i * ((tc + 273.15)**4 - T_p_K**4)
                termo_irradiacao = termo_Qi / Q_F
                
                numerador = (1 - termo_irradiacao) * eta_F * PCI + \
                            AC * Cp_ar_F * (t_aq_estimado - T_amb_C) + \
                            C_cb * (t_comb - T_amb_C) - \
                            f_F * p5_atual * PCI
                
                # Usando a variável 'z' do PDF para a fração de cinzas
                z = cinzas_fracao 
                denominador = (AC + 1 - z) * Cp_g_atual
                
                tc_calculado = (numerador / denominador) + T_amb_C
                return tc_calculado - tc
    
            # Resolve a equação não-linear para encontrar um novo tc
            tc_solucao = fsolve(equacao_fornalha, [tc_estimado_loop])[0]
            
            # Verifica a convergência e atualiza o chute se necessário
            if abs(tc_solucao - tc_estimado_loop) < 0.1: break
            tc_estimado_loop = tc_solucao
            
        # Define o valor final de tc como o último valor convergido
        tc_final = tc_estimado_loop
        
        # "Então", como diz o PDF, calculamos Qi e sigma com o tc_final
        Q_i_final = beta * emis_C * S_i * ((tc_final + 273.15)**4 - T_p_K**4)
        Q_F_final = eta_F * m_dot_cb_atual * PCI
        sigma_final = Q_i_final / Q_F_final if Q_F_final > 0 else 0
    
        return tc_final, Q_i_final, sigma_final
    
    def calcular_superaquecedor_simulacao(
        # --- Parâmetros fixos (constantes da simulação) ---
        S_S,                  # Área do superaquecedor (do projeto nominal) [m²]
        h_s_nom,              # Entalpia nominal na saída (h_s do seu script) [kcal/kg]
        h_v_nom,              # Entalpia do vapor saturado (precisa ser calculada no nominal) [kcal/kg]
        cps,                  # Calor específico médio (calculado antes da simulação)
        T_superaq,            # Temperatura alvo do superaquecedor (do cenário) [°C]
        T_v_C,                # Temperatura de saturação (calculada no nominal) [°C]
        
        # --- Parâmetros gerais ---
        AC,                   # Relação ar/combustível
        z,                    # Fração de cinzas
        f_S,                  # Fator de perda do superaquecedor (ex: 0.15)
        Cp_0C, Cp_1200C,
        
        # --- Parâmetros variáveis (do regime/iteração atual) ---
        tc_final,             # Temperatura de saída da fornalha (resultado do Passo 1) [°C]
        m_dot_cb_atual,       # Consumo de combustível do regime atual [kg/h]
        m_v_estimado,         # Vazão de vapor estimada na iteração [kg/h]
        p5_atual,             # Perda p5 corrigida para o regime atual
        U_s_atual,            # Coeficiente U corrigido para o superaquecedor
        PCI                   # PCI do combustível
        ):
        """
        Passo 2 da Simulação: Calcula tS e tSR para o Superaquecedor.
        Resolve o sistema 2x2 de equações e a dependência de Cpg.
        """
    
        # Chutes iniciais para as temperaturas que queremos encontrar
        tsr_estimado_loop = tc_final - 100 # Chute para a temperatura de saída dos gases
        ts_estimado_chute = T_superaq      # Chute para a temperatura do vapor (valor nominal)
    
        for i in range(10): # Loop para convergir Cpg e tSR
            T_media_gases = (tc_final + tsr_estimado_loop) / 2.0
            Cp_g_atual = Cp_0C + (Cp_1200C - Cp_0C) * (T_media_gases / 1200)
    
            def sistema_superaquecedor(vars):
                ts, tsr = vars
                
                # Equação para o calor absorvido pelo vapor
                Q_S_calc = m_v_estimado * (h_s_nom - h_v_nom + cps * (ts - T_superaq))
    
                # Residual 1: Balanço de Energia
                Q5_componente = p5_atual * m_dot_cb_atual * PCI * f_S
                calor_gases = m_dot_cb_atual * (AC + 1 - z) * Cp_g_atual * (tc_final - tsr) - Q5_componente
                residual1 = Q_S_calc - calor_gases
                
                # Residual 2: Transferência de Calor
                delta_t1 = tc_final - ts
                delta_t2 = tsr - T_v_C
                
                if delta_t1 <= 0 or delta_t2 <= 0 or abs(delta_t1 - delta_t2) < 1e-6:
                    return [1e6, 1e6]
                
                delta_t_ls = (delta_t1 - delta_t2) / np.log(delta_t1 / delta_t2)
                calor_conveccao = U_s_atual * S_S * delta_t_ls
                residual2 = Q_S_calc - calor_conveccao
    
                return [residual1, residual2]
    
            try:
                ts_solucao, tsr_solucao = fsolve(sistema_superaquecedor, [ts_estimado_chute, tsr_estimado_loop])
            except Exception:
                # Em caso de erro, retorna os chutes para não parar a simulação
                return ts_estimado_chute, tsr_estimado_loop
    
            if abs(tsr_solucao - tsr_estimado_loop) < 0.1: break
            tsr_estimado_loop = tsr_solucao
    
        ts_final = ts_solucao
        tsr_final = tsr_estimado_loop
        
        return ts_final, tsr_final
    
    def calcular_reaquecedor_simulacao(
        # --- Parâmetros fixos (constantes da simulação) ---
        S_R,                  # Área do reaquecedor (do projeto nominal) [m²]
        cpr,                  # Calor específico médio (calculado antes da simulação)
        T_reaq,               # Temperatura alvo do reaquecedor (do cenário) [°C]
        T_e_C,                # Temperatura de saturação na entrada do reaquecedor [°C]
        h_r_reaq_nom,         # Entalpia nominal na saída do reaquecedor [kcal/kg]
        h_e_reaq_nom,         # Entalpia nominal na entrada do reaquecedor [kcal/kg]
        m_E_fracao,           # Fração mássica de reaquecimento (do cenário)
        
        # --- Parâmetros gerais ---
        AC,                   # Relação ar/combustível
        z,                    # Fração de cinzas
        f_R,                  # Fator de perda do reaquecedor (ex: 0.15)
        Cp_0C, Cp_1200C,
        
        # --- Parâmetros variáveis (do regime/iteração atual) ---
        tsr_final,            # Temperatura de saída do superaquecedor (resultado do Passo 2) [°C]
        m_dot_cb_atual,       # Consumo de combustível do regime atual [kg/h]
        m_v_estimado,         # Vazão de vapor estimada na iteração [kg/h]
        p5_atual,             # Perda p5 corrigida para o regime atual
        U_r_atual,            # Coeficiente U corrigido para o reaquecedor
        PCI
        ):
        """
        Passo 3 da Simulação: Calcula tR e tRV para o Reaquecedor.
        Resolve o sistema 2x2 de equações e a dependência de Cpg.
        """
    
        # Chutes iniciais para as temperaturas que queremos encontrar
        trv_estimado_loop = tsr_final - 100 # Chute para a temperatura de saída dos gases
        tr_estimado_chute = T_reaq          # Chute para a temperatura do vapor (valor nominal)
    
        for i in range(10): # Loop para convergir Cpg e tRV
            T_media_gases = (tsr_final + trv_estimado_loop) / 2.0
            Cp_g_atual = Cp_0C + (Cp_1200C - Cp_0C) * (T_media_gases / 1200)
    
            def sistema_reaquecedor(vars):
                tr, trv = vars
                
                # Equação para o calor absorvido pelo vapor
                Q_R_calc = m_E_fracao * m_v_estimado * (h_r_reaq_nom - h_e_reaq_nom + cpr * (tr - T_reaq))
    
                # Residual 1: Balanço de Energia
                Q5_componente = p5_atual * m_dot_cb_atual * PCI * f_R
                calor_gases = m_dot_cb_atual * (AC + 1 - z) * Cp_g_atual * (tsr_final - trv) - Q5_componente
                residual1 = Q_R_calc - calor_gases
                
                # Residual 2: Transferência de Calor
                delta_t1 = tsr_final - tr
                delta_t2 = trv - T_e_C
                
                if delta_t1 <= 0 or delta_t2 <= 0 or abs(delta_t1 - delta_t2) < 1e-6:
                    return [1e6, 1e6]
                
                delta_t_lr = (delta_t1 - delta_t2) / np.log(delta_t1 / delta_t2)
                calor_conveccao = U_r_atual * S_R * delta_t_lr
                residual2 = Q_R_calc - calor_conveccao
    
                return [residual1, residual2]
    
            try:
                tr_solucao, trv_solucao = fsolve(sistema_reaquecedor, [tr_estimado_chute, trv_estimado_loop])
            except Exception:
                return tr_estimado_chute, trv_estimado_loop
    
            if abs(trv_solucao - trv_estimado_loop) < 0.1: break
            trv_estimado_loop = trv_solucao
    
        tr_final = tr_solucao
        trv_final = trv_estimado_loop
        
        return tr_final, trv_final
    
    def calcular_vaporizador_simulacao(
        # --- Parâmetros fixos (constantes da simulação) ---
        S_V,                  # Área do vaporizador (do projeto nominal) [m²]
        c,                    # Calor específico médio da água no economizador [kcal/kg°C]
        T_v_C,                # Temperatura de saturação (calculada no nominal) [°C]
        h_v_nom,              # Entalpia do vapor saturado (h_v) [kcal/kg]
        h_1_nom,              # Entalpia da água de alimentação (h_1) [kcal/kg]
        theta1,               # Temperatura da água de alimentação (do cenário) [°C]
    
        # --- Parâmetros gerais ---
        AC,                   # Relação ar/combustível
        z,                    # Fração de cinzas
        f_V,                  # Fator de perda do vaporizador (ex: 0.1)
        Cp_0C, Cp_1200C,
        
        # --- Parâmetros variáveis (do regime/iteração atual) ---
        trv_final,            # Temperatura de saída do reaquecedor (resultado do Passo 3) [°C]
        Qi_final,             # Calor irradiado, resultado da função da Fornalha [kcal/h]
        m_dot_cb_atual,       # Consumo de combustível do regime atual [kg/h]
        p5_atual,             # Perda p5 corrigida para o regime atual
        U_v_atual,            # Coeficiente U corrigido para o vaporizador
        theta2_estimado,      # A temperatura 'theta2' estimada para a iteração [°C]
        PCI
        ):
        """
        Passo 4 da Simulação: Calcula tVE, Qv e a nova vazão de vapor (m_v).
        """
    
        # Chute inicial para tVE
        tve_estimado_loop = trv_final - 100
        
        # Loop para convergir Cpg e tVE
        for i in range(10):
            T_media_gases = (trv_final + tve_estimado_loop) / 2.0
            Cp_g_atual = Cp_0C + (Cp_1200C - Cp_0C) * (T_media_gases / 1200)
    
            def equacao_vaporizador(vars):
                tve = vars[0]
                
                Q5_componente = p5_atual * m_dot_cb_atual * PCI * f_V
                calor_gases = m_dot_cb_atual * (AC + 1 - z) * Cp_g_atual * (trv_final - tve) - Q5_componente
                
                delta_t1 = trv_final - T_v_C
                delta_t2 = tve - T_v_C
    
                if delta_t1 <= 0 or delta_t2 <= 0 or abs(delta_t1 - delta_t2) < 1e-6:
                    return 1e6
    
                delta_t_lv = (delta_t1 - delta_t2) / np.log(delta_t1 / delta_t2)
                calor_conveccao = U_v_atual * S_V * delta_t_lv
                
                return calor_gases - calor_conveccao
    
            try:
                tve_solucao = fsolve(equacao_vaporizador, [tve_estimado_loop])[0]
            except Exception:
                tve_solucao = tve_estimado_loop
    
            if abs(tve_solucao - tve_estimado_loop) < 0.1: break
            tve_estimado_loop = tve_solucao
            
        tve_final = tve_estimado_loop
        
        # "Então", calculamos o calor Qv com o tve_final convergido
        Q5_comp_final = p5_atual * m_dot_cb_atual * PCI * f_V
        Q_v_final = m_dot_cb_atual * (AC + 1 - z) * Cp_g_atual * (trv_final - tve_final) - Q5_comp_final
    
        # E finalmente, recalculamos a vazão de vapor m_v
        # Usando a aproximação h2 = h1 + c*(theta2 - theta1)
        h2_aproximado = h_1_nom + c * (theta2_estimado - theta1)
        denominador_mv = h_v_nom - h2_aproximado
        
        if denominador_mv <= 0:
            print("  ERRO: Denominador nulo ou negativo ao calcular m_v.")
            m_v_calculado = 0 # Retorna 0 em caso de erro
        else:
            m_v_calculado = (Q_v_final + Qi_final) / denominador_mv
    
        return tve_final, Q_v_final, m_v_calculado
    
    def calcular_economizador_simulacao(
        # --- Parâmetros fixos (constantes da simulação) ---
        S_E,                  # Área do economizador (do projeto nominal) [m²]
        c,                    # Calor específico médio da água no economizador [kcal/kg°C]
        theta1,               # Temperatura da água de alimentação (do cenário) [°C]
    
        # --- Parâmetros gerais ---
        AC,                   # Relação ar/combustível
        z,                    # Fração de cinzas
        f_E,                  # Fator de perda do economizador (ex: 0.1)
        Cp_0C, Cp_1200C,
        
        # --- Parâmetros variáveis (do regime/iteração atual) ---
        tve_final,            # Temperatura de saída do vaporizador (resultado do Passo 4) [°C]
        m_v_calculado,        # Vazão de vapor, recalculada no Passo 4 [kg/h]
        m_dot_cb_atual,       # Consumo de combustível do regime atual [kg/h]
        p5_atual,             # Perda p5 corrigida para o regime atual
        U_e_atual,            # Coeficiente U corrigido para o economizador
        PCI
        ):
        """
        Passo 5 da Simulação: Calcula theta2 e tEA para o Economizador.
        """
    
        # Chutes iniciais para as temperaturas
        tea_estimado_loop = tve_final - 100
        theta2_estimado_chute = theta1 + 20 # Um chute razoável
    
        for i in range(10): # Loop para convergir Cpg e tEA
            T_media_gases = (tve_final + tea_estimado_loop) / 2.0
            Cp_g_atual = Cp_0C + (Cp_1200C - Cp_0C) * (T_media_gases / 1200)
    
            def sistema_economizador(vars):
                theta2, tea = vars
                
                # Equação para o calor absorvido pela água
                Q_E_calc = m_v_calculado * c * (theta2 - theta1)
    
                # Residual 1: Balanço de Energia
                Q5_componente = p5_atual * m_dot_cb_atual * PCI * f_E
                calor_gases = m_dot_cb_atual * (AC + 1 - z) * Cp_g_atual * (tve_final - tea) - Q5_componente
                residual1 = Q_E_calc - calor_gases
                
                # Residual 2: Transferência de Calor
                delta_t1 = tve_final - theta2
                delta_t2 = tea - theta1
                
                if delta_t1 <= 0 or delta_t2 <= 0 or abs(delta_t1 - delta_t2) < 1e-6:
                    return [1e6, 1e6]
                
                delta_t_le = (delta_t1 - delta_t2) / np.log(delta_t1 / delta_t2)
                calor_conveccao = U_e_atual * S_E * delta_t_le
                residual2 = Q_E_calc - calor_conveccao
    
                return [residual1, residual2]
    
            try:
                theta2_solucao, tea_solucao = fsolve(sistema_economizador, [theta2_estimado_chute, tea_estimado_loop])
            except Exception:
                return theta2_estimado_chute, tea_estimado_loop
    
            if abs(tea_solucao - tea_estimado_loop) < 0.1: break
            tea_estimado_loop = tea_solucao
    
        theta2_final = theta2_solucao
        tea_final = tea_estimado_loop
        
        return theta2_final, tea_final
    
    def calcular_aquecedor_ar_simulacao(
        # --- Parâmetros fixos (constantes da simulação) ---
        S_A,                  # Área do aquecedor de ar (do projeto nominal) [m²]
        T_amb_C,                # Temperatura do ar de entrada (ambiente) [°C]
    
        # --- Parâmetros gerais ---
        AC,                   # Relação ar/combustível
        z,                    # Fração de cinzas
        f_A,                  # Fator de perda do aquecedor de ar (ex: 0.1)
        Cp_0C, Cp_1200C,
        
        # --- Parâmetros variáveis (do regime/iteração atual) ---
        tea_final,            # Temperatura de saída do economizador (resultado do Passo 5) [°C]
        m_dot_cb_atual,       # Consumo de combustível do regime atual [kg/h]
        p5_atual,             # Perda p5 corrigida para o regime atual
        U_a_atual,            # Coeficiente U corrigido para o aquecedor de ar
        PCI
        ):
        """
        Passo 6 da Simulação: Calcula taq e tg para o Aquecedor de Ar.
        """
    
        # Chutes iniciais para as temperaturas
        tg_estimado_loop = tea_final - 150
        taq_estimado_loop = 250
    
        for i in range(10): # Loop para convergir Cpg, Cpar, taq e tg
            # Calcular Cpg (gases) com base na estimativa de tg
            T_media_gases = (tea_final + tg_estimado_loop) / 2.0
            Cp_g_atual = Cp_0C + (Cp_1200C - Cp_0C) * (T_media_gases / 1200)
    
            # Calcular Cpar (ar) com base na estimativa de taq
            T_media_ar = (taq_estimado_loop + T_amb_C) / 2.0
            Cp_par_atual = Cp_0C + (Cp_1200C - Cp_0C) * (T_media_ar / 1200)
    
            def sistema_aquecedor_ar(vars):
                taq, tg = vars
                
                # Calor total transferido para o componente (aquece o ar + perdas)
                Q_A_calc = m_dot_cb_atual * AC * Cp_par_atual * (taq - T_amb_C) + (p5_atual * m_dot_cb_atual * PCI * f_A)
    
                # Residual 1: Balanço de Energia
                calor_gases = m_dot_cb_atual * (AC + 1 - z) * Cp_g_atual * (tea_final - tg)
                residual1 = Q_A_calc - calor_gases
                
                # Residual 2: Transferência de Calor (com correção do typo do PDF Ue->Ua, Se->Sa)
                delta_t1 = tea_final - taq
                delta_t2 = tg - T_amb_C
                
                if delta_t1 <= 0 or delta_t2 <= 0 or abs(delta_t1 - delta_t2) < 1e-6:
                    return [1e6, 1e6]
                
                delta_t_la = (delta_t1 - delta_t2) / np.log(delta_t1 / delta_t2)
                calor_conveccao = U_a_atual * S_A * delta_t_la
                residual2 = Q_A_calc - calor_conveccao
    
                return [residual1, residual2]
    
            try:
                taq_solucao, tg_solucao = fsolve(sistema_aquecedor_ar, [taq_estimado_loop, tg_estimado_loop])
            except Exception:
                return taq_estimado_loop, tg_estimado_loop
    
            if (abs(taq_solucao - taq_estimado_loop) < 0.1 and abs(tg_solucao - tg_estimado_loop) < 0.1): break
                
            taq_estimado_loop = taq_solucao
            tg_estimado_loop = tg_solucao
            
        taq_final = taq_estimado_loop
        tg_final = tg_estimado_loop
        
        return taq_final, tg_final
# ... e assim por diante para Reaquecedor, Vaporizador, Economizador e Aquecedor de Ar.

    # ===================================================================================
    # FASE 4: EXECUÇÃO DA SIMULAÇÃO
    # ===================================================================================
    print("\n" + "="*60)
    print("FASE 4: INICIANDO SIMULAÇÃO PARA DIFERENTES REGIMES")
    print("="*60)
    
    # Lista para armazenar o dicionário de resultados de cada regime convergido
    resultados_finais = []
    # Define os regimes de operação a serem simulados
    regimes_R = np.arange(30, 121, 10)
    
    # --- Chutes iniciais para o PRIMEIRO regime de operação ---
    # Usamos os valores nominais como ponto de partida
    ts_estimado = T_SR_C_nom
    tr_estimado = T_RV_C_nom
    theta2_estimado = theta2_C_nom
    taq_estimado = T_ar_aq_C # Do seu código de projeto original
    # Chutes para temperaturas intermediárias dos gases (valores razoáveis)
    tsr_estimado = ts_estimado + 50 
    trv_estimado = tr_estimado + 50
    tve_estimado = 400
    tea_estimado = 300
    tg_estimado = 160
    
    # --- Loop Principal Externo (para cada regime de operação) ---
    for R_perc in regimes_R:
        print(f"\n--- CALCULANDO PARA O REGIME DE {R_perc}% ---")
        
        # a) Calcular parâmetros que dependem diretamente do regime
        m_dot_cb_atual = m_dot_cb_nom * (R_perc / 100)
        p5_atual = p5_nom * (100 / R_perc)**0.9 if R_perc > 0 else p5_nom
        
        # Corrigir os coeficientes U para o regime atual
        U_s_atual = U_s_nom * (m_dot_cb_atual / m_dot_cb_nom)**0.62
        U_r_atual = U_r_nom * (m_dot_cb_atual / m_dot_cb_nom)**0.62
        U_v_atual = U_v_nom * (m_dot_cb_atual / m_dot_cb_nom)**0.62
        U_e_atual = U_e_nom * (m_dot_cb_atual / m_dot_cb_nom)**0.62
        U_a_atual = U_a_nom * (m_dot_cb_atual / m_dot_cb_nom)**0.86
        
        Q_F_atual = eta_F * m_dot_cb_atual * PCI
        Q_5_atual = p5_atual * m_dot_cb_atual * PCI
        m_v_estimado = m_v_nom * R_perc/100     
    
        # b) "Grande Iteração" (loop interno para convergir o sistema)
        for i in range(50): # Limite de 50 iterações para evitar loop infinito
            
            # Guardar valores da iteração anterior para checar a convergência
            m_v_antigo = m_v_estimado
            ts_antigo, tr_antigo, theta2_antigo, taq_antigo = ts_estimado, tr_estimado, theta2_estimado, taq_estimado
    
            # --- c) Chamar as funções dos componentes em sequência ---
            
            # 1. Fornalha
            tc_final, Qi_final, sigma_final = calcular_fornalha_simulacao(
                S_i=S_i_nom, T_p_K=T_p_K, eta_F=eta_F, AC=AC, PCI=PCI, emis_C=emis_C,
                cinzas_fracao=z, t_comb=T_cb_C, C_cb=C_cb, f_F=f_F, T_amb_C=T_amb_C,
                Cp_ar_F=Cp_ar_F, Cp_0C=Cp_0C, Cp_1200C=Cp_1200C,
                m_dot_cb_atual=m_dot_cb_atual, p5_atual=p5_atual, t_aq_estimado=taq_estimado
            )
                
            # 2. Superaquecedor
            ts_final, tsr_final = calcular_superaquecedor_simulacao(
                S_S=S_S_nom, h_s_nom=h_s_nom, h_v_nom=h_v_nom, cps=cps, T_superaq=T_SR_C, 
                T_v_C=T_v_C_nom, AC=AC, z=z, f_S=f_S, Cp_0C=Cp_0C, Cp_1200C=Cp_1200C,
                tc_final=tc_final, m_dot_cb_atual=m_dot_cb_atual, m_v_estimado=m_v_estimado,
                p5_atual=p5_atual, U_s_atual=U_s_atual, PCI=PCI
            )
            
            # 3. Reaquecedor
            tr_final, trv_final = calcular_reaquecedor_simulacao(
                S_R=S_R_nom, cpr=cpr, T_reaq=T_RV_C, T_e_C=T_e_C_nom, h_r_reaq_nom=h_r_nom,
                h_e_reaq_nom=h_e_nom, m_E_fracao=m_e, AC=AC, z=z, f_R=f_R, Cp_0C=Cp_0C,
                Cp_1200C=Cp_1200C, tsr_final=tsr_final, m_dot_cb_atual=m_dot_cb_atual,
                m_v_estimado=m_v_estimado, p5_atual=p5_atual, U_r_atual=U_r_atual, PCI=PCI
            )
            
            # 4. Vaporizador (calcula a nova vazão de vapor)
            tve_final, Qv_final, m_v_calculado = calcular_vaporizador_simulacao(
               S_V=S_V_nom, c=c, T_v_C=T_v_C_nom, h_v_nom=h_v_nom, h_1_nom=h_1_nom,
               theta1=theta1_C,AC=AC, z=z, f_V=f_V, Cp_0C=Cp_0C, Cp_1200C=Cp_1200C,
               trv_final=trv_final, Qi_final=Qi_final, m_dot_cb_atual=m_dot_cb_atual,
               p5_atual=p5_atual, U_v_atual=U_v_atual, theta2_estimado=theta2_estimado, PCI=PCI
            )
            
            # 5. Economizador
            theta2_final, tea_final = calcular_economizador_simulacao(
                S_E=S_E_nom, c=c, theta1=theta1_C,  # <-- CORRIGIDO: Usando a variável correta 'theta1_C'
                AC=AC, z=z, f_E=f_E, Cp_0C=Cp_0C, Cp_1200C=Cp_1200C,
                tve_final=tve_final, m_v_calculado=m_v_calculado,
                m_dot_cb_atual=m_dot_cb_atual, p5_atual=p5_atual, U_e_atual=U_e_atual, PCI=PCI
            )
        
            # 6. Aquecedor de Ar
            taq_final, tg_final = calcular_aquecedor_ar_simulacao(
                S_A=S_A_nom, T_amb_C=T_amb_C, AC=AC, z=z, f_A=f_A, Cp_0C=Cp_0C,
                Cp_1200C=Cp_1200C, tea_final=tea_final, m_dot_cb_atual=m_dot_cb_atual,
                p5_atual=p5_atual, U_a_atual=U_a_atual, PCI=PCI
    )
    
            # --- d) Atualizar as estimativas para a próxima iteração ---
            m_v_estimado = m_v_calculado
            ts_estimado = ts_final
            tr_estimado = tr_final
            theta2_estimado = theta2_final
            taq_estimado = taq_final
            # Atualizar chutes das temperaturas intermediárias também
            tsr_estimado, trv_estimado, tve_estimado, tea_estimado, tg_estimado = tsr_final, trv_final, tve_final, tea_final, tg_final
            
            # --- e) Checar a convergência da "Grande Iteração" ---
            erro_mv = abs((m_v_estimado - m_v_antigo) / m_v_antigo) if m_v_antigo > 0 else 1.0
            erro_ts = abs(ts_estimado - ts_antigo)
            erro_tr = abs(tr_estimado - tr_antigo)
            erro_theta2 = abs(theta2_estimado - theta2_antigo)
            erro_taq = abs(taq_estimado - taq_antigo)
            
            # Critérios do PDF: ±0.5% para vazão de vapor, ±1°C para temperaturas
            if erro_mv < 0.005 and erro_ts < 1.0 and erro_tr < 1.0 and erro_theta2 < 1.0 and erro_taq < 1.0:
                print(f"  --> Sistema convergiu na iteração {i+1} para {R_perc}%.")
                
                # --- f) Cálculos finais após a convergência do regime ---
                T_media_gases_chamine = (tg_final + T_amb_C) / 2.0
                Cpg_chamine_final = Cp_0C + (Cp_1200C - Cp_0C) * (T_media_gases_chamine / 1200)
                p6_final = (m_dot_cb_atual * (AC + 1 - z) * Cpg_chamine_final * (tg_final - T_amb_C)) / (m_dot_cb_atual * PCI)
                eta_Final = 1 - (P1 + P2 + P3 + P4 + p5_atual + p6_final)
    
                # --- g) Armazenar os resultados finais ---
                resultado_regime = {
                    'regime_%': R_perc,
                    'm_cb_kg_h': m_dot_cb_atual,
                    'm_v_kg_h': m_v_estimado,
                    'rendimento': eta_Final * 100, # em %
                    't_c': tc_final,
                    't_s': ts_final,
                    't_r': tr_final,
                    'theta2': theta2_final,
                    't_aq': taq_final,
                    't_g': tg_final
                }
                resultados_finais.append(resultado_regime)
                
                # Sair do loop da "Grande Iteração"
                break
            
    else: # Este 'else' pertence ao 'for i' e executa se o break não ocorrer
        print(f"  ALERTA: Sistema não convergiu para {R_perc}% após 50 iterações.")
    
    # ===================================================================================
    # FASE 5: ANÁLISE E PLOTAGEM DOS RESULTADOS FINAIS
    # ===================================================================================
    print("\n\n" + "="*60)
    print("FASE 5: RESULTADOS FINAIS DA SIMULAÇÃO")
    print("="*60)
    
    df_resultados = pd.DataFrame(resultados_finais)
    print(df_resultados)
    
    # Chamar a função para gerar os gráficos
    # gerar_graficos_operacionais(df_resultados)