import random
import math
import pygame as pg
import tensorflow_text as text
import tensorflow as tf

model = tf.keras.models.load_model('sprakmodell.h5')

# skaffer vokabularet vi har laget

bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
bert_vocab_args = dict(
    # maksimum størrelse for vokabularet
    vocab_size=8000 * 7,
    # Reserverte orddeler som må være med
    reserved_tokens=reserved_tokens,
    # flere argumenter
    bert_tokenizer_params=bert_tokenizer_params,
    learn_params={},
)

# lager en "tokenizer", som deler tekst opp i orddeler
tokenizer = text.BertTokenizer('vocab.txt', **bert_tokenizer_params)

tokenlist = open('vocab.txt', 'r', encoding="utf-8").readlines()


# IDen til padding
PAD_ID = 0
# Maksimum lengde for vektoren
# hvis vektoren er mindre, blir det lagt til padding
max_seq_len = 20
langs = 7


def preprocess_bert_input(text):
    # finner IDene til alle orddelene i inputtet
    ids = tokenize_text(text, max_seq_len)
    # lager en mask, som i dette tilfettet representerer lengden på vektoren vår
    mask = tf.cast(ids > 0, tf.int64)
    mask = tf.reshape(mask, [-1, max_seq_len])
    # lager den ferdige vektoren
    # først fyller lager vi en vektor med
    # den riktige lengden (shape) fyllt med nuller
    zeros_dims = tf.stack(tf.shape(mask))
    type_ids = tf.fill(zeros_dims, PAD_ID)
    # så setter vi inn de faktiske orddelenes IDer
    type_ids = tf.cast(type_ids, tf.int64)

    return (ids, mask, type_ids)


def tokenize_text(text, seq_len):
    # bruker "tokenizeren" vi lagde tidligere til å generere tokens som passer teksten
    tokens = tokenizer.tokenize(text)
    # tilpasser outputtet
    tokens = tokens.merge_dims(1, 2)[:, :seq_len]

    # klipper vekk slutten hvis den er lenger enn maksimum lengde
    tokens = tokens[:, :seq_len]
    # legger til padding hvis den er kortere enn maksimum lengde
    tokens = tokens.to_tensor(default_value=PAD_ID)
    pad = seq_len - tf.shape(tokens)[1]
    tokens = tf.pad(tokens, [[0, 0], [0, pad]], constant_values=PAD_ID)
    return tf.reshape(tokens, [-1, seq_len])


def run_model(text):
    # deler inputtet inn i orddeler
    t = preprocess_bert_input(text)
    # kjører modellen på inputtet
    return model.predict([t])[0]


class Body:
    def __init__(self, inputs, pos):
        self.weights = [0 for i in range(langs)]
        for inp in inputs:
            result = run_model(inp)
            for i in range(langs):
                self.weights[i] += result[i]
        for i in range(langs):
            self.weights[i] /= langs

        self.pos = pos
        self.vel = [0, 0]

    def affect(self, positions):

        force = [0, 0]
        for i in range(langs):
            dist = math.sqrt((positions[i][0] - self.pos[0])**2 +
                             (positions[i][1] - self.pos[1])**2)
            if dist < 0.01:
                continue

            goal_dist = (1 - self.weights[i]) * 400

            factor = ((dist - goal_dist) * 0.00005)

            force[0] += (positions[i][0] - self.pos[0]) * factor
            force[1] += (positions[i][1] - self.pos[1]) * factor

        self.vel[0] += force[0]
        self.vel[1] += force[1]

        self.vel[0] *= 0.8
        self.vel[1] *= 0.8

        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]


def main():
    main_bodies = []
    width, height = 1000, 576

    def random_pos():
        return [random.randrange(-width / 2, width / 2), random.randrange(-height / 2, height / 2)]

    # da
    main_bodies.append(Body([
        "deres komplekse struktur muliggør en langt bredere vifte af udtryk end noget kendt system af dyrs kommunikation",
        "den indiske økonomi er den fjerdestørste i verden set i forhold til købekraften",
        "i den tid bredte betegnelsen canada sig – et navn der oprindelig var et irokesisk stednavn",
        "det grænser op til papua ny guinea indonesien og østtimor mod nord salomonøerne og vanuatu mod nordøst og new zealand mod sydøst",
        "med en befolkning på knap 14 milliard mennesker hvilket er ca"
    ], random_pos()))

    # en
    main_bodies.append(Body([
        "when mathematical structures are good models of real phenomena mathematical reasoning can be used to provide insight or predictions about nature",
        "in its application across business problems machine learning is also referred to as predictive analytics",
        "in the punjab sikhism emerged rejecting institutionalised religion",
        "it ranks among the highest in international measurements of government transparency civil liberties quality of life economic freedom and education",
        "since then china has expanded fractured and reunified numerous times"
    ], random_pos()))

    # es
    main_bodies.append(Body([
        "por su parte los animales se comunican a través de signos sonoros olfativos y corporales que en muchos casos distan de ser sencillos",
        "wikipedia ha recibido diversas críticas",
        "limita con el océano índico al sur con el mar arábigo al oeste y con el golfo de bengala al este a lo largo de una línea costera de más de 7517 kilómetros",
        "a causa de su clima es uno de los 15 países con menor densidad poblacional del mundo con aproximadamente 4 habitantes por kilómetro cuadrado",
        "durante milenios su sistema político se basó en monarquías hereditarias conocidas como dinastías"
    ], random_pos()))

    # ja
    main_bodies.append(Body([
        "tyuu 2 hondo igai no tiiki de wa kore to kotonaru zikantai o saiyou site iru mono mo aru",
        "igirisu renpou kameikoku de ari ei renpou oukoku no itikoku to natte iru",
        "taiheiyou sensougo no 1947 nen ni wa genkou no nipponkoku kenpou o sikou",
        "gaikou ni oite wa 1956 nen kara kokusai rengou ni kamei site ori kokuren tyuusin syugi o totte iru",
        "sono tame puroguramingu no purosesu ni wa apurikeesyon domein ni kansuru tisiki tokutei no arugorizumu keisiki ronri nado samazama na syudai ni kansuru senmonsei ga youkyuu sareru koto ga ooi"
    ], random_pos()))

    # nb
    main_bodies.append(Body([
        "språk kan også henvise til bruken av slike systemer som et generelt fenomen",
        "alle norske universitet og enkelte høgskoler tilbyr studier som inkluderer maskinlæring som tema",
        "den nest største utgaven er den nederlandske med mer enn 18 millioner artikler",
        "de sentrale delene av landet har fruktbart sletteland prærie kontinentalt klima og lite nedbør",
        "landets historie kjennetegnes av gjentatte delinger og gjenforeninger mellom skiftende perioder av fred og krig og voldelige dynastiske skifter"
    ], random_pos()))

    # nn
    main_bodies.append(Body([
        "hovudstaden i india er new delhi",
        "det bur litt meir enn 20 millionar innbyggjarar i australia",
        "hovudstad og administrasjonsstad er berlin",
        "ved hjelp av programmeringsspråk kan ein person laga instruksar til ein datamaskin som fortel den korleis den skal løyse ei gitt oppgåve",
        "fagfeltet studerer teknikkar som gjer maskina i stand til å lære altså på eiga hand kunne forbetre evna si til problemløysing"
    ], random_pos()))

    # sv
    main_bodies.append(Body([
        "maskininlärningsmetoder arbetar med data",
        "kanada är med en areal om 9985 miljoner kvadratkilometer världens näst största land till ytan efter ryssland",
        "man har sedan dess haft ett stabilt liberalt demokratiskt politiskt system som fungerar som en federal parlamentarisk demokrati och konstitutionell monarki",
        "kina är en ekonomisk och militär stormakt och har ett permanent säte i fns säkerhetsråd",
        "sedan man antog sin reviderade konstitution 1947 har japan haft en enhetlig konstitutionell monarki med en kejsare och ett folkvalt parlament"
    ], random_pos()))

    for i in range(langs):
        for j in range(langs):
            avg = (main_bodies[i].weights[j] + main_bodies[j].weights[i]) / 2
            main_bodies[i].weights[j] = avg
            main_bodies[j].weights[i] = avg

    done = False

    screen = pg.display.set_mode(
        (width, height), pg.RESIZABLE)  # pg.FULLSCREEN
    font = pg.font.Font(None, 32)
    clock = pg.time.Clock()

    class_names = ['dansk', 'engelsk', 'spansk',
                   'japansk', 'bokmål', 'nynorsk', 'svensk']

    colors = ['#ff2424', '#083e9c', '#e3da36',
              '#ffffff', '#5966f7', '#ff2424', '#fcf630']

    tbw, tbh = 300, 32
    textbox = pg.Rect(width / 2 - tbw / 2, height * 0.1 - tbh / 2, tbw, tbh)
    text = ''

    input_body = Body([], [0, 0])

    while not done:
        width, height = screen.get_size()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_BACKSPACE:
                    text = text[:-1]
                elif event.key == pg.K_ESCAPE:
                    done = True
                    continue
                elif event.key == pg.K_UP:
                    paste = pg.scrap.get("text/plain;charset=utf-8")
                    if paste:
                        text = str(paste, encoding="utf-8")
                else:
                    text += event.unicode
                text = text.lower()
                input_body.weights = run_model(text)
        screen.fill((0, 0, 0))

        # Render the current text.
        txt_surface = font.render(text, True, pg.Color('white'))
        # Resize the box if the text is too long.
        tb_width = max(tbw, txt_surface.get_width()+10)
        textbox.w = tb_width
        textbox.x = width / 2 - tb_width / 2
        # Blit the text.
        screen.blit(txt_surface, (textbox.x+5, textbox.y+5))
        # Blit the input_box rect.
        pg.draw.rect(screen, pg.Color('white'), textbox, 2)

        positions = []
        for i in range(langs):
            positions.append(main_bodies[i].pos)

        for i in range(langs):
            main_bodies[i].affect(positions)

        input_body.affect(positions)

        avg_pos = [0, 0]
        for i in range(langs):
            avg_pos[0] += main_bodies[i].pos[0]
            avg_pos[1] += main_bodies[i].pos[1]
        avg_pos[0] += input_body.pos[0]
        avg_pos[1] += input_body.pos[1]
        avg_pos[0] /= langs + 1
        avg_pos[1] /= langs + 1

        r = 10

        for i in range(langs):
            for j in range(langs):
                line_col = pg.Color(50, 50, 50)

                pg.draw.line(
                    screen,
                    line_col,
                    (width / 2 + main_bodies[i].pos[0] - avg_pos[0],
                     height / 2 + main_bodies[i].pos[1] - avg_pos[1]),
                    (width / 2 + main_bodies[j].pos[0] - avg_pos[0],
                     height / 2 + main_bodies[j].pos[1] - avg_pos[1]),
                    round(main_bodies[i].weights[j] * 10)
                )

        for j in range(langs):
            line_col = pg.Color(100, 50, 100)

            pg.draw.line(
                screen,
                line_col,
                (width / 2 + input_body.pos[0] - avg_pos[0],
                    height / 2 + input_body.pos[1] - avg_pos[1]),
                (width / 2 + main_bodies[j].pos[0] - avg_pos[0],
                    height / 2 + main_bodies[j].pos[1] - avg_pos[1]),
                round(input_body.weights[j] * 10)
            )

        for i in range(langs):

            rect = pg.Rect(
                width / 2 + main_bodies[i].pos[0] - r - avg_pos[0],
                height / 2 + main_bodies[i].pos[1] -
                r - avg_pos[1], r * 2, r * 2
            )
            # print(rect)

            pg.draw.ellipse(screen, pg.Color(colors[i]), rect, 2)
            txt_surface = font.render(
                class_names[i], True, pg.Color(colors[i]))

            screen.blit(txt_surface, (rect.x + r -
                                      txt_surface.get_width() - 20, rect.y + r - txt_surface.get_height() / 2))

        # input node
        rect = pg.Rect(
            width / 2 + input_body.pos[0] - r - avg_pos[0],
            height / 2 + input_body.pos[1] -
            r - avg_pos[1], r * 2, r * 2
        )
        # print(rect)

        pg.draw.ellipse(screen, pg.Color('white'), rect, 2)
        txt_surface = font.render(
            'input', True, pg.Color('white'))

        screen.blit(txt_surface, (rect.x + r -
                                  txt_surface.get_width() - 20, rect.y + r - txt_surface.get_height() / 2))

        pg.display.flip()
        clock.tick(30)


pg.init()

pg.scrap.init()
main()
pg.quit()
