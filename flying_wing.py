# =============================================================================
# flying_wing.py
# =============================================================================
# Nova classe FlyingWing, herdeira de SimpleAircraft.
#
# O que muda em relação ao SimpleAircraft original:
#
#   REMOVIDO:  aileron(), elevator(), rudder()  — três superfícies independentes
#   ADICIONADO: elevon_left(), elevon_right()   — cada elevon age nos dois eixos
#               elevon(delta_pitch, delta_roll) — interface única de comando
#
# POR QUÊ herdar de SimpleAircraft e não de RigidBody diretamente?
#
#   SimpleAircraft já tem toda a física aerodinâmica pronta:
#   apply_drag(), apply_lift(), apply_aero_torque(), apply_angular_drag(),
#   apply_thrust(), drain_fuel(), update_throttle().
#   Não precisamos reescrever nada disso — só substituímos os métodos
#   de controle. Herança é exatamente para isso.
#
# CONCEITO DE ELEVON — resumo rápido:
#
#   Em uma aeronave convencional:
#       Profundor (elevator) → controla PITCH (arfagem, nariz sobe/desce)
#       Aileron              → controla ROLL  (rolamento, asa sobe/desce)
#
#   Em uma asa voadora, as duas funções são fundidas em uma única superfície
#   por lado (elevon = elevator + aileron):
#
#       δ_pitch = comando de arfagem  (positivo = nariz sobe)
#       δ_roll  = comando de rolamento (positivo = asa esquerda sobe)
#
#       Elevon ESQUERDO: δL = δ_pitch + δ_roll
#       Elevon DIREITO:  δR = δ_pitch - δ_roll
#
#   Exemplo — só pitch (cabrar):
#       δ_pitch = +1, δ_roll = 0  →  δL = +1,  δR = +1  (ambos sobem igual)
#
#   Exemplo — só roll (rolar para a direita):
#       δ_pitch = 0, δ_roll = +1  →  δL = +1,  δR = -1  (diferencial)
#
#   Exemplo — pitch + roll simultâneos:
#       δ_pitch = +0.5, δ_roll = +0.8  →  δL = +1.3 (limitado a +1),  δR = -0.3
#
# =============================================================================

import numpy as np
from rigidbody import SimpleAircraft   # importa a classe pai


class FlyingWing(SimpleAircraft):
    """
    Asa voadora com superfícies de controle do tipo elevon.

    Parâmetros adicionais em relação ao SimpleAircraft:
        effectiveness_pitch : float
            Efetividade do elevon no eixo de pitch (torque por unidade
            de deflexão por (m/s)²). Equivale a control_effectiveness[1]
            do SimpleAircraft original.

        effectiveness_roll : float
            Efetividade do elevon no eixo de roll (torque por unidade
            de deflexão por (m/s)²). Equivale a control_effectiveness[0].

    Parâmetros que NÃO existem mais:
        control_effectiveness[2] (rudder) — asa voadora não tem leme convencional.
        Se a aeronave tiver winglets com split drag rudder, adicione um método
        separado drag_rudder(direction) seguindo o mesmo padrão.
    """

    def __init__(self, model, CoM,
                 pos, vel, accel,
                 orient, ang_vel, ang_accel,
                 mass, inertia,
                 max_thrust, throttle_range, throttle,
                 prop_mass, mass_flow,
                 cross_sections, Cds, Cdas, angular_damping, Cl,
                 effectiveness_pitch, effectiveness_roll):

        # ------------------------------------------------------------------
        # Chamamos o __init__ do SimpleAircraft, passando control_effectiveness
        # como um array de zeros — assim o objeto pai é construído corretamente,
        # mas sem nenhuma efetividade de controle "embutida" nas superfícies
        # antigas (aileron/elevator/rudder), que não vamos usar.
        # ------------------------------------------------------------------
        control_effectiveness_dummy = np.array([0.0, 0.0, 0.0])

        super().__init__(model, CoM,
                         pos, vel, accel,
                         orient, ang_vel, ang_accel,
                         mass, inertia,
                         max_thrust, throttle_range, throttle,
                         prop_mass, mass_flow,
                         cross_sections, Cds, Cdas, angular_damping, Cl,
                         control_effectiveness_dummy)

        # Guardamos as efetividades reais dos elevons como atributos próprios.
        self.effectiveness_pitch = effectiveness_pitch
        self.effectiveness_roll  = effectiveness_roll

        # Estado interno dos elevons: deflexões atuais de cada um.
        # Úteis para visualização futura (ex: renderizar a superfície inclinada)
        # e para log de telemetria.
        self.delta_left  = 0.0   # deflexão atual do elevon esquerdo  [-1, +1]
        self.delta_right = 0.0   # deflexão atual do elevon direito   [-1, +1]


    # ==========================================================================
    # MÉTODOS DE CONTROLE DOS ELEVONS
    # ==========================================================================

    def elevon_left(self, delta):
        """
        Aplica o torque gerado pelo elevon ESQUERDO.

        O elevon esquerdo age simultaneamente em dois eixos:

          PITCH: deflexão positiva (borda de fuga desce) gera nariz para cima.
                 O torque é no eixo X do corpo (orient[0]).
                 Sinal: +delta → +torque_pitch

          ROLL:  deflexão positiva da superfície esquerda levanta a asa
                 esquerda → roll para a DIREITA.
                 O torque é no eixo Z do corpo (orient[2]).
                 Sinal: +delta → -torque_roll
                 (negativo porque asa esquerda subindo = roll anti-horário
                 visto de trás = positivo no eixo Z depende da convenção;
                 aqui usamos a mesma do aileron original do Kelaynak,
                 onde +Z = roll para a esquerda, então esquerda subindo = -Z)

        Parâmetro:
            delta : float em [-1, +1]
                Deflexão do elevon esquerdo (já calculada pela mixagem).
        """
        vel_sq = np.linalg.norm(self.vel) ** 2
        # A força aerodinâmica de uma superfície cresce com v² — igual a
        # qualquer força aerodinâmica (F = ½ρv²·S·C). Como no original.

        # Componente de PITCH (eixo X do corpo):
        torque_pitch = np.array([1.0, 0.0, 0.0]) * delta * vel_sq * self.effectiveness_pitch

        # Componente de ROLL (eixo Z do corpo):
        # O elevon ESQUERDO subindo (delta > 0) gera roll para a DIREITA,
        # ou seja, torque negativo no eixo Z (convenção do Kelaynak).
        torque_roll  = np.array([0.0, 0.0, 1.0]) * (-delta) * vel_sq * self.effectiveness_roll

        self.apply_torque(torque_pitch + torque_roll)
        self.delta_left = delta   # salva para visualização/telemetria


    def elevon_right(self, delta):
        """
        Aplica o torque gerado pelo elevon DIREITO.

        PITCH: igual ao esquerdo — deflexão positiva → nariz para cima.
               +delta → +torque_pitch  (mesmo sinal)

        ROLL:  deflexão positiva da superfície direita levanta a asa
               direita → roll para a ESQUERDA.
               +delta → +torque_roll  (sinal oposto ao esquerdo)

        Parâmetro:
            delta : float em [-1, +1]
                Deflexão do elevon direito (já calculada pela mixagem).
        """
        vel_sq = np.linalg.norm(self.vel) ** 2

        # Componente de PITCH — mesmo sinal que o esquerdo:
        torque_pitch = np.array([1.0, 0.0, 0.0]) * delta * vel_sq * self.effectiveness_pitch

        # Componente de ROLL — sinal OPOSTO ao esquerdo:
        # Elevon DIREITO subindo (delta > 0) → asa direita sobe → roll para a esquerda
        # → torque positivo no eixo Z.
        torque_roll  = np.array([0.0, 0.0, 1.0]) * delta * vel_sq * self.effectiveness_roll

        self.apply_torque(torque_pitch + torque_roll)
        self.delta_right = delta


    def elevon(self, delta_pitch, delta_roll):
        """
        Interface principal de controle: recebe os comandos do piloto
        em termos intuitivos (pitch e roll) e faz a MIXAGEM para os elevons.

        Esta é a função que o loop do main.py deve chamar a cada frame,
        substituindo as três chamadas:
            AP.aileron(ctrl_state[0])
            AP.elevator(ctrl_state[1])
            AP.rudder(ctrl_state[2])
        por uma única chamada:
            AP.elevon(ctrl_state[1], ctrl_state[0])
                       ↑ pitch          ↑ roll

        Parâmetros:
            delta_pitch : float em [-1, +1]
                Comando de arfagem. +1 = nariz sobe (cabrar). -1 = nariz desce (picar).

            delta_roll : float em [-1, +1]
                Comando de rolamento. +1 = rolar para a ESQUERDA (asa esquerda sobe).
                -1 = rolar para a DIREITA.

        MIXAGEM:
            δL = δ_pitch + δ_roll   (elevon esquerdo)
            δR = δ_pitch - δ_roll   (elevon direito)

            Por que esse sinal?
            - Se só pitch (+1, 0):  δL = +1, δR = +1  → ambos sobem igual → nariz sobe ✓
            - Se só roll (+1):      δL = +1, δR = -1  → esquerda sobe, direita desce → roll ✓
            - Pitch + roll:         ambos os efeitos se combinam ✓

        Clipping em [-1, +1]:
            A soma pode ultrapassar o limite físico da superfície
            (ex: δ_pitch = +0.8, δ_roll = +0.8 → δL = +1.6, impossível mecanicamente).
            np.clip() limita ao intervalo válido, como fazem os transmissores RC reais.
        """
        delta_L = np.clip(delta_pitch + delta_roll, -1.0, 1.0)
        delta_R = np.clip(delta_pitch - delta_roll, -1.0, 1.0)

        self.elevon_left(delta_L)
        self.elevon_right(delta_R)


    # ==========================================================================
    # MÉTODO HERDADO — SOBRESCRITO PARA DESABILITAR
    # ==========================================================================
    # Os três métodos abaixo são sobrescritos para não fazer nada.
    # Isso garante que, se por acidente o código antigo do main.py chamar
    # AP.aileron() ou AP.elevator(), nada vai acontecer de errado.
    # É uma proteção contra chamadas acidentais.

    def aileron(self, direction):
        """Desabilitado: asa voadora não tem aileron. Use elevon()."""
        pass

    def elevator(self, direction):
        """Desabilitado: asa voadora não tem profundor. Use elevon()."""
        pass

    def rudder(self, direction):
        """
        Desabilitado: asa voadora não tem leme convencional.
        Se a aeronave tiver split drag rudder nos winglets, implemente
        um método drag_rudder(direction) separado aqui.
        """
        pass
