# =============================================================================
# main.py — KELAYNAK FLIGHT SIMULATOR
# =============================================================================
# Este é o arquivo principal do simulador. Ele é responsável por:
#   1. Criar e configurar todos os objetos do mundo (aeronave, terreno, câmera…)
#   2. Rodar o loop principal do simulador (o "coração" que repete ~60x por segundo)
#   3. A cada repetição: ler o teclado → calcular física → renderizar a cena → repetir
# obs: as coordenadas [x, y, z] são x horizontal apontando para um dos lados, 
# y vertical apontando para cima, z horizontal apontando para frente
# =============================================================================

# importações necessárias de bibliotecas
import numpy as np
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import time
import random
import keyboard as kbd

# importa móduloes internos do simulador kelaynak 
from rigidbody import *
from model import *         # carrega modelos 3D (.mdl)
from graphics import *
from camera import *
from terrain import *
from ui import *
from scenery_objects import *
from sound import *
from alerts import *

def main():                # função principal, onde reside o simulador
    
    def window_resize(window, width, height):        # função para redimensionar a janela
        try:
            # glfw.get_framebuffer_size(window)
            glViewport(0, 0, width, height)
            glLoadIdentity()
            gluPerspective(fov, width/height, near_clip, far_clip)
            glTranslate(main_cam.pos[0], main_cam.pos[1], main_cam.pos[2])
            main_cam.orient = np.eye(3)
            main_cam.rotate([0, 180, 0])
        except ZeroDivisionError:
            # if the window is minimized it makes height = 0, but we don't need to update projection in that case anyway
            pass

    # INIT VESSELS
    print("Initializing vessels...")
    
    #------------------- Bloco foguete (ignorar) ----------------------
    #condições iniciais para o FOGUETE (posição, velocidade, aceleração, orientação, velocidade angular,
    #aceleração angular, massa, inercia, impulso máximo, range do acelerador, acelerador,
    # massa propulsiva (foguete), fluxo de massa (foguete))
    init_pos = np.array([0.0, 100.0, 0.0])          # m
    init_vel = np.array([0.0, 20.0, 0.0])            # m s-1
    init_accel = np.array([0.0, 0.0, 0.0])          # m s-2
    init_orient = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
    init_ang_vel = np.array([0.0, 0.0, 0.0])        # rad s-1
    init_ang_accel = np.array([0.0, 0.0, 0.0])      # rad s-2
    init_mass = 1000                                # kg
    init_inertia = np.array([[5.0, 0.0, 0.0],
                             [0.0, 5.0, 0.0],
                             [0.0, 0.0, 10.0]])     # kg m2
    max_thrust = 20e3                               # N
    throttle_range = [60, 100]                      # %
    throttle = 100                                  # %
    prop_mass = 800                                 # kg
    mass_flow = 3                                   # kg s-1

    rocket_model = Model("rocket")
    init_CoM = np.array([0.0, 2.5, 0.0])

    # criação do objeto foguete de nome KS
    KS = Rocket(rocket_model, init_CoM,
                init_pos, init_vel, init_accel,
                init_orient, init_ang_vel, init_ang_accel,
                init_mass, init_inertia,
                max_thrust, throttle_range, throttle,
                prop_mass, mass_flow)
    #--------------------- fim bloco foguete -----------------------
    
    # condições iniciais para AERONAVE. é o principal a ser alterado
    # posição começa a 2000 m de altitude:
    init_pos = np.array([0.0, 2000.0, 0.0])         # m
    # velocidade em x, y, z. estpa começando em 100 m/s para frente (em z):
    init_vel = np.array([0.0, 0.0, 100.0])          # m s-1
    # aceleração inicial:
    init_accel = np.array([0.0, 0.0, 0.0])          # m s-2
    # orientação inicial:
    init_orient = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
    # velocidade angular inicial:
    init_ang_vel = np.array([0.0, 0.0, 0.0])        # rad s-1
    # aceleração angular inicial:
    init_ang_accel = np.array([0.0, 0.0, 0.0])      # rad s-2
    # massa inicial:
    init_mass = 800                                 # kg
    # inércia inicial (eixo x ou roll, eixo y ou pitch, eixo z ou yaw) em kg/m2:
    #  Todos os produtos de inércia (Ixy, Ixz, Iyz) são zero — a matriz é diagonal.
    # Isso assume simetria perfeita. Na realidade, Ixz ≠ 0 em asas voadoras (segundo claude)
    init_inertia = np.array([[6000.0, 0.0, 0.0],
                             [0.0, 6000.0, 0.0],
                             [0.0, 0.0, 3000.0]])   # kg m2
    # empuxo máximo em N:
    max_thrust = 5e3                               # N
    # throttle pode ir de 0% a 100%:
    throttle_range = [0, 100]                       # %
    # throttle inicial:
    throttle = 100                                  # %
    # massa de combustível propulsiva:
    prop_mass = 150                                 # kg
    # fluxo de massa de 0,001kg/s (perda de massa):
    # no caso de motor elétrico, não há perda de massa. verificar como mass_flow = 0 pode afetar o restante do código
    mass_flow = 0.001                                # kg s-1

    plane_model = Model("plane_cockpit")            # carrega o modelo 3d do cockpit de /data/models
    init_CoM = np.array([0.0, 0.0, 2.0])            # centro de massa inicial (CG)

     # Parâmetros aerodinâmicos 
    # seções transversais em m2 nos três eixos do corpo:
    cross_sections = np.array([8, 10, 2])            # m2
    # coeficientes de arrasto (cd):
    # arrasto lateral (deriva), arrasto vertical (mergulho/subida abrupta), arrasto frontal 
    Cds = np.array([0.6, 0.8, 0.1])
    # coeficientes de arrasto angular (resistência a rotação em cada eixo):
    # resistência ao rolamento (roll), à arfagem (pitch), à guinada (yaw)
    Cdas = np.array([8, 20, 45])
    # Amortecimento angular DIRETO aplicado sobre a velocidade angular ω:
    # A cada frame: ω[i] = ω[i] × (1 − angular_damping[i] × dt)
    # Isso "freia" artificialmente a rotação para simular efeitos aerodinâmicos
    # complexos que o modelo de corpo único não consegue reproduzir.
    angular_damping = np.array([0.4, 0.8, 0.8])
    # Coeficiente de sustentação global:
    # Multiplica diretamente a força de sustentação calculada: F_lift = mult_AoA × Cl × 0,5 × cross_y × v².
    # Não é o CL clássico (que depende do AoA explicitamente via curva polar) — aqui é um escalar fixo que amplifica a sustentação.
    Cl = 1.2
    
    # efetividade das superfícies de controle
    # aileron (rolamento), elevator/ profundor (arfagem), rudder/leme (guinada)
    # O torque real é: T = control_effectiveness[i] × ctrl_state[i] × |v|² (proporcional ao quadrado da velocidade), definido em rigidbody.py
    control_effectiveness = np.array([1.8, 1.8, 2.5])

    # cria o objeto avião de nome AP com os parâmetros definidos acima
    # a classe SimpleAircraft (definida em rigidbody.py) herda de RigidBody e adiciona a lógica aerodinâmica
    AP = SimpleAircraft(plane_model, init_CoM,
                        init_pos, init_vel, init_accel,
                        init_orient, init_ang_vel, init_ang_accel,
                        init_mass, init_inertia,
                        max_thrust, throttle_range, throttle,
                        prop_mass, mass_flow,
                        cross_sections, Cds, Cdas, angular_damping, Cl,
                        control_effectiveness)
    AP.set_thrust_percent(80)            # reduz o throttle inicial de 100% (definido anteriormente em throttle) para 80%. motor parcial

    # corpo ativo na simulação (aeronave AP)
    bodies = [AP]

    # SCENERY OBJECTS
    # criação do cenário
    print("Initializing scenery objects...")
    pylon_model = Model("pylon")        # coloca modelos pylon (mastros de referência) de /data/models, da classe model.py
    pylon1 = SceneryObject(pylon_model, np.array([10,0,500])) # 10 m à direita, 500 m à frente
    pylon2 = SceneryObject(pylon_model, np.array([-10,0,500])) # 10 m à esquerda, 500 m à frente

    # scenery_objects = [pylon1, pylon2] # pylons estão comentados
    scenery_objects = []    # cenário inicia sem objetos (pylons comentados) e depois recebe prédios aleatórios

    # RANDOM BUILDINGS
    # gera prédios aleatórios em uma área quadrada de 5km x 5km centrado na origem
    Nx = 100 
    Nz = 100
    chance = 0.01
    building_spacing_x = 50
    building_spacing_z = 50

    building_area_corner_x = Nx / 2 * building_spacing_x
    building_area_corner_z = Nz / 2 * building_spacing_z

    # essa linha não é utilizada. os prédios são colocados diretamente em scenery_objects
    buildings = []

    # gera os prédios aleatoriamente
    for idx_x in range(Nx):
        for idx_z in range(Nz):
            if random.uniform(0, 1) < chance:
                c_x = -building_area_corner_x + idx_x * building_spacing_x
                c_z = -building_area_corner_z + idx_z * building_spacing_z
                new_pos = np.array([c_x, 0, c_z])
                new_building = RandomBuilding(new_pos)
                scenery_objects.append(new_building)

    # TERRAIN
    # cria um chão plano na altura y=0 
    print("Initializing terrain...")
    floor = Flatland(0, Color(0.1, 0.8, 0.1)) # A cor é RGB (0.1, 0.8, 0.1) — verde.

    # MISC PHYSICS
    # aceleração da gravidade (negativa - ou para baixo - em y). Alterar para gravidade em são josé dos campos
    gravity = np.array([0.0, -9.81, 0])

    # GRAPHICS
    # inicializa os gráficos
    print("Initializing graphics (OpenGL, glfw)...")
    window_x, window_y = 1600, 900 # resolução da janela em pixels
    fov = 70 # field of view (campo de visão) de 70 graus (60-90, maior = mais 'olho de peixe')
    near_clip = 0.1 # distância mínima de renderização (m)
    far_clip = 10e6 # distância máxima de renderização (m)
    
    glfw.init()
    window = glfw.create_window(window_x, window_y, "Kelaynak Flight Simulator", None, None)
    glfw.set_window_pos(window, 100, 100)
    glfw.make_context_current(window)
    glfw.set_window_size_callback(window, window_resize)

    gluPerspective(fov, window_x/window_y, near_clip, far_clip)
    glClearColor(0, 0, 0.3, 1) # Define a cor de fundo da janela: RGB(0, 0, 0.3) = azul escuro, opacidade 1.

    # SOUND
    print("Initializing sound (pygame.mixer)...")
    init_sound()

    # CAMERA
    # inicialização da câmera
    cam_pos = np.array([0, 0, 0]) 
    cam_orient = np.array([[-1, 0, 0],
                           [0, 1, 0],
                           [0, 0, -1]]) # O eixo X negativo e Z negativo reflete a diferença de convenção entreo sistema de coordenadas do OpenGL (Z aponta para o usuário) e o do simulador (Z aponta para frente)
    main_cam = Camera("main_cam", cam_pos, cam_orient, True)

    glRotate(-180, 0, 1, 0)    
    main_cam.lock_to_target(bodies[0]) # trava a câmera no avião

    def move_cam(movement):
        main_cam.move(movement)

    def rotate_cam(rotation):
        main_cam.rotate(rotation)

    # CAMERA CONTROLS
    # controles da câmera
    cam_pitch_up = "K"    # Câmera: girar para cima   
    cam_pitch_dn = "I"    # Câmera: girar para baixo
    cam_yaw_left = "J"    # Câmera: girar para a esquerda
    cam_yaw_right = "L"   # Câmera: girar para a direita
    cam_roll_cw = "O"     # Câmera: girar no sentido horário
    cam_roll_ccw = "U"    # Câmera: girar no sentido anti-horário

    cam_move_fwd   = "Y"  # Câmera: mover para frente
    cam_move_bck   = "H"  # Câmera: mover para trás
    cam_move_left  = "N"  # Câmera: mover para a esquerda
    cam_move_right = "M"  # Câmera: mover para a direita
    cam_move_up    = "Y"  # NOTA: mesma tecla que cam_move_fwd ("Y") — provável bug
    cam_move_dn    = "H"  # NOTA: mesma tecla que cam_move_bck ("H") — provável bug

    # controles do avião
    plane_pitch_up  = "S"  # Profundor: nariz para cima (cabrar)
    plane_pitch_dn  = "W"  # Profundor: nariz para baixo (picar)
    plane_roll_ccw  = "Q"  # Aileron: rolar para a esquerda
    plane_roll_cw   = "E"  # Aileron: rolar para a direita
    plane_yaw_right = "D"  # Leme: guinada para a direita
    plane_yaw_left  = "A"  # Leme: guinada para a esquerda
    plane_throttle_up = "Z"  # Throttle: aumentar potência
    plane_throttle_dn = "X"  # Throttle: reduzir potência

    # modo primeira pessoa
    first_person_ui = True

    cam_speed = 100 # velocidade de translação da câmera livre m/s
    cam_rot_speed = 100 # velocidade de rotação da câmera livre graus/s

    # inicialização dos sons contínuos
    play_sfx("turbojet_fan", -1, 1, 0)
    play_sfx("wind1", -1, 2, 0)
    play_sfx("rumble", -1, 3, 0)

    # Variáveis de estado do loop principal
    print("Starting...")
    # dt (delta time) = duração do frame anterior, em segundos.
    # É a variável mais importante do simulador: ela conecta "o que acontece por segundo" com "o que acontece neste frame específico". ex: veloc += acel x dt
    dt = 0
    # estado atual dos três controles [aileron, elevator, rudder]. vai de -1 a 1 (deflexão máxima negativa a máxima positiva)
    ctrl_state = [0, 0, 0]

    # fatores de conversão de unidades para o HUD
    velocity_conversion_factor = 1
    altitude_conversion_factor = 1

    # LOOP PRINCIPAL
    # roda até o usuário fechar a janela. tudo do simulador ocorre dentro desse loop. até antes era só inicialização
    # faz:
    #   1. Marcar início do frame
    #   2. Processar eventos do sistema operacional
    #   3. Ler teclado e atualizar controles
    #   4. Calcular física
    #   5. Verificar colisão com o solo
    #   6. Atualizar câmera
    #   7. Renderizar cena
    #   8. Montar e exibir HUD
    #   9. Atualizar volumes dos sons
    #   10. Emitir alertas
    #   11. Calcular dt do frame que acabou de terminar
    while not glfw.window_should_close(window):
        t_cycle_start = time.perf_counter()
        glfw.poll_events() 

        # CONTROLS
        # leitura do teclado --------------------------------------------------------------
        if kbd.is_pressed(cam_move_fwd):
            move_cam([0, 0, cam_speed * dt])
        if kbd.is_pressed(cam_move_bck):
            move_cam([0, 0, -cam_speed * dt])
        if kbd.is_pressed(cam_move_up):
            move_cam([0, -cam_speed * dt, 0])
        if kbd.is_pressed(cam_move_dn):
            move_cam([0, cam_speed * dt, 0])
        if kbd.is_pressed(cam_move_right):
            move_cam([-cam_speed * dt, 0, 0])
        if kbd.is_pressed(cam_move_left):
            move_cam([cam_speed * dt, 0, 0])

        if kbd.is_pressed(cam_pitch_up):
            rotate_cam([cam_rot_speed * dt, 0, 0])
        if kbd.is_pressed(cam_pitch_dn):
            rotate_cam([-cam_rot_speed * dt, 0, 0])
        if kbd.is_pressed(cam_yaw_left):
            rotate_cam([0, cam_rot_speed * dt, 0])
        if kbd.is_pressed(cam_yaw_right):
            rotate_cam([0, -cam_rot_speed * dt, 0])
        if kbd.is_pressed(cam_roll_cw):
            rotate_cam([0, 0, cam_rot_speed * dt])
        if kbd.is_pressed(cam_roll_ccw):
            rotate_cam([0, 0, -cam_rot_speed * dt])

        # leitura do teclado para controles do avião ---------------------------------------
        # o controle de pitch é o mais elaborado: ctrl_state[1] aumenta ou diminui gradualmente 
        # quando a tecla é pressionada e retorna suavemente a zero quando a tecla é solta
        if kbd.is_pressed(plane_pitch_up):
            ctrl_state[1] += 1 * dt # pressionando S, a deflexão de profundor é aumentada em 1 unidade por segundo (chega a 1,0 em 1s)
        elif kbd.is_pressed(plane_pitch_dn):
            ctrl_state[1] -= 1 * dt # com W pressionado, diminui a deflexão (nariz para baixo)
        else: # sem nenhuma tecla pressionada, retorna ao neutro como uma exponencial
            if abs(ctrl_state[1]) > 0.3:
                ctrl_state[1] *= 1 - 2 * dt 
            else:
                ctrl_state[1] = 0

        # roll (aileron): mesmo padrão gradual e suavizado
        if kbd.is_pressed(plane_roll_ccw):
            ctrl_state[0] += 1 * dt
        elif kbd.is_pressed(plane_roll_cw):
            ctrl_state[0] -= 1 * dt
        else:
            if abs(ctrl_state[0]) > 0.3:
                ctrl_state[0] *= 1 - 2 * dt
            else:
                ctrl_state[0] = 0

        # yaw (leme): mesmo padrão gradual e suavizado
        if kbd.is_pressed(plane_yaw_right):
            ctrl_state[2] += 1 * dt
        elif kbd.is_pressed(plane_yaw_left):
            ctrl_state[2] -= 1 * dt
        else:
            if abs(ctrl_state[2]) > 0.3:
                ctrl_state[2] *= 1 - 2 * dt
            else:
                ctrl_state[2] = 0

        # throttle: 
        if kbd.is_pressed(plane_throttle_up):
            AP.update_throttle(30, dt) # tecla Z aumenta o throttle em 30%/s
        elif kbd.is_pressed(plane_throttle_dn):
            AP.update_throttle(-30, dt) # tecla X diminui o throttle em 30%/s

        for i in range(len(ctrl_state)):
            ctrl_state[i] = min(max(ctrl_state[i], -1), 1)

        # aplica comandos de controle como torques na aeronave
        AP.aileron(ctrl_state[0])  # Gera torque no eixo Z (roll) proporcional a ctrl_state[0] × |v|²
        AP.elevator(ctrl_state[1]) # Gera torque no eixo X (pitch) proporcional a ctrl_state[1] × |v|²
        AP.rudder(ctrl_state[2])   # Gera torque no eixo Y (yaw) proporcional a ctrl_state[2] × |v|²

        # troca de unidades no HUD (display)
        if kbd.is_pressed("M"): # superior metric units for the superior people
            velocity_conversion_factor = 1
            altitude_conversion_factor = 1
        elif kbd.is_pressed("N"): # inferior imperial units for Mars Climate Orbiter
            velocity_conversion_factor = 1.943844 # knots
            altitude_conversion_factor = 3.28084 # feet

        # PHYSICS
        # cálculo da física ---------------------------------------------------------------
        # parte mais importante. Usa funções de rigidbody.py
        # A ordem de aplicação das forças importa porque todas são acumuladas em self.accel e self.ang_accel
        AP.drain_fuel(dt)         # Diminui a massa de combustível: prop_mass -= mass_flow × throttle% × dt. se o combustível acabar, o empuxo é zerado
        AP.apply_aero_torque()    # calcula e acumula torques de arfagem e guindada causados pela velocidade perpendicular ao eixo da aeronave (resistência aerodinâmica rotacional)
        AP.apply_angular_drag(dt) # calcula e acumula o arrasto angular (resistência à rotação), aplica amortecimento direto sobre w
        AP.apply_drag()           # calcula e acumula forças de arrasto translacional em todos os eixos
        AP.apply_lift()           # calcula e acumula a força de sustentação baseada no AoA atual
        AP.apply_thrust()         # acumula a força de empuxo na direção longitudinal da aeronave (orient[2])
        G = np.linalg.norm(AP.accel) / 10 # calcula o fator G antes de adicionar a gravidade
        AP.apply_accel(gravity)   # adiciona a aceleração da gravidade à aceleração acumulada
        AP.update(dt)             # aplica a aceleração acumulada à velocidade, a velocidade à posição, a aceleração angular à velocidade angular,
        # e rotaciona a matriz de orientação. Depois zera as acelerações para o próximo frame.

        # calcula o AoA (ângulo de ataque) para a exibição no HUD (display) e para os alertas
        AoA = np.arccos(max(min(np.dot(AP.vel, AP.orient[2]) / np.linalg.norm(AP.vel), 1), -1))
        AoA = np.rad2deg(AoA)
        # np.dot(AP.vel, AP.orient[2]) = produto interno entre a velocidade e o eixo longitudinal da aeronave → cos(AoA).Dividir pela norma da velocidade normaliza para [-1, 1].
        # max(min(...)) garante que o valor esteja em [-1,1] antes do arccos, evitando erros numéricos de ponto flutuante (ex: 1.0000001).
        # np.rad2deg converte o resultado de radianos para graus.

        # hit flat ground
        # colisão com o solo --------------------------------------------------------------
        # reposiciona aeronave no nível do solo se y<0
        for b in bodies:
            if b.pos[1] < floor.height:
                b.pos[1] = 0
                b.vel[1] = 0
                b.vel = b.vel - b.vel * 0.05 * dt # aplica atrito simplificado no solo (aproximação grosseira)

        # atualização da câmera -----------------------------------------------------------
        main_cam.move_with_lock(dt)
        main_cam.rotate_with_lock(dt)

        # GRAPHICS
        # renderização da cena 3D ---------------------------------------------------------
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        drawScene(main_cam, floor, bodies, scenery_objects, ctrl_state, first_person_ui)

        # HUD (display) - instrumentos na tela --------------------------------------------
        alt_string = "Alt: " + str(int(AP.pos[1] * altitude_conversion_factor))
        vel_string = "Vel: " + str(int(np.linalg.norm(AP.vel) * velocity_conversion_factor))
        throttle_str = "Throttle: " + str(int(AP.throttle))
        AoA_str = "AOA: " + str(round(AoA, 2))
        G_str = "G: " + str(round(G, 2))

        magenta = Color(1, 0, 1)
        red = Color(1, 0, 0)
        
        AoA_color = magenta
        G_color = magenta
        vel_color = magenta
        alt_color = magenta
        throttle_color = magenta

        if np.linalg.norm(AP.vel) < 50 and AoA > 10:
            vel_color = red
            AoA_color = red
            if AP.throttle < 100:
                throttle_color = red

        if G > 9:
            G_color = red

        if AP.pos[1] < 1000 and AP.vel[1] < 0 and AP.pos[1] / -AP.vel[1] < 3:
            alt_color = red
        
        render_AN(alt_string, alt_color, [4, 5], main_cam, fpu=first_person_ui)
        render_AN(vel_string, vel_color, [-7, 5], main_cam, fpu=first_person_ui)
        render_AN(throttle_str, throttle_color, [-7, -4.5], main_cam, fpu=first_person_ui)
        render_AN(AoA_str, AoA_color, [-7, -5.5], main_cam, fpu=first_person_ui)
        render_AN(G_str, G_color, [-7, -5], main_cam, fpu=first_person_ui)
        
        glfw.swap_buffers(window)

        # atualização dos volumes de áudio ---------------------------------------------------
        set_channel_volume(1, AP.throttle / 100 * 0.5) # engine
        set_channel_volume(2, min(np.linalg.norm(AP.vel) / 500, 1) * 0.5) # airflow
        set_channel_volume(3, min(G / 10, 1) * 0.5) # airflow disturbance

        # alertas sonoros --------------------------------------------------------------------
        do_warnings(AP, AoA, G)

        # cálculo de dt (tempo do frame) -----------------------------------------------------
        dt = time.perf_counter() - t_cycle_start
        # LIMITAÇÃO IMPORTANTE: o dt inclui tanto o tempo de cálculo físico
        # quanto o de renderização gráfica. Se a GPU travar por um frame,
        # o próximo frame avança mais na física — pode causar instabilidade.
        # A solução correta seria separar os loops de física e gráficos em
        # threads independentes com dt fixo para a física.

    # encerramento do simulador
    glfw.destroy_window(window)
    stop_channel(1)
    stop_channel(2)
    stop_channel(3)

# chama a função principal
main()
