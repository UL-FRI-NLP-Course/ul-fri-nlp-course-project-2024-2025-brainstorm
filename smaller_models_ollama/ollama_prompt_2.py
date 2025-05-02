from ollama import chat
from ollama import ChatResponse
import time

pravila_za_formulacijo = """
LJUBLJANA-KOPER – PRIMORSKA AVTOCESTA/ proti Kopru/proti Ljubljani
LJUBLJANA-OBREŽJE – DOLENJSKA AVTOCESTA / proti Obrežju/ proti Ljubljani
LJUBLJANA-KARAVANKE – GORENJSKA AVTOCESTA/ proti Karavankam ali Avstriji/ proti Ljubljani
LJUBLJANA-MARIBOR – ŠTAJERSKA AVTOCESTA / proti Mariboru/Ljubljani
MARIBOR-LENDAVA – POMURSKA AVTOCESTA / proti Mariboru/ proti Lendavi/Madžarski
MARIBOR-GRUŠKOVJE – PODRAVSKA AVTOCESTA / proti Mariboru/ proti Gruškovju ali Hrvaški – nikoli proti Ptuju!
AVTOCESTNI ODSEK – RAZCEP GABRK – FERNETIČI – proti Italiji/ ali proti primorski avtocesti, Kopru, Ljubljani (PAZI: to ni primorska avtocesta)
AVTOCESTNI ODSEK MARIBOR-ŠENTILJ (gre od mejnega prehoda Šentilj do razcepa Dragučova) ni štajerska avtocesta kot pogosto navede PIC, ampak je avtocestni odsek od Maribora proti Šentilju oziroma od Šentilja proti Mariboru.
Mariborska vzhodna obvoznica= med razcepom Slivnica in razcepom Dragučova – smer je proti Avstriji/Lendavi ali proti Ljubljani – nikoli proti Mariboru. 

Hitre ceste skozi Maribor uradno ni več - Ni BIVŠA hitra cesta skozi Maribor, ampak regionalna cesta Betnava-Pesnica oziroma NEKDANJA hitra cesta skozi Maribor.

Ljubljanska obvoznica je sestavljena iz štirih krakov= vzhodna, zahodna, severna in južna 
Vzhodna: razcep Malence (proti Novemu mestu) - razcep Zadobrova (proti Mariboru) 
Zahodna: razcep Koseze (proti Kranju) – razcep Kozarje (proti Kopru)
Severna: razcep Koseze (proti Kranju) – razcep Zadobrova (proti Mariboru)
Južna: razcep Kozarje (proti Kopru) – razcep Malence (proti Novemu mestu)
Hitra cesta razcep Nanos-Vrtojba = vipavska hitra cesta – proti Italiji ali Vrtojbi/ proti Nanosu/primorski avtocesti/proti Razdrtemu/v smeri Razdrtega (nikoli primorska hitra cesta – na Picu večkrat neustrezno poimenovanje) 
Hitra cesta razcep Srmin-Izola – obalna hitra cesta – proti Kopru/Portorožu (nikoli primorska hitra cesta)
Hitra cesta Koper-Škofije (manjši kos, poimenuje kar po krajih): Na hitri cesti od Kopra proti Škofijam ali obratno na hitri cesti od Škofij proti Kopru – v tem primeru imaš notri zajeto tudi že smer. (nikoli primorska hitra cesta). Tudi na obalni hitri cesti od Kopra proti Škofijam.
Hitra cesta mejni prehod Dolga vas-Dolga vas: majhen odsek pred mejnim prehodom, formulira se navadno kar na hitri cesti od mejnega prehoda Dolga vas proti pomurski avtocesti; v drugo smer pa na hitri cesti proti mejnemu prehodu Dolga vas – zelo redko v uporabi. 
Regionalna cesta: ŠKOFJA LOKA – GORENJA VAS (= pogovorno škofjeloška obvoznica) – proti Ljubljani/proti Gorenji vasi. Pomembno, ker je velikokrat zaprt predor Stén.
GLAVNA CESTA Ljubljana-Črnuče – Trzin : glavna cesta od Ljubljane proti Trzinu/ od Trzina proti Ljubljani – včasih vozniki poimenujejo  trzinska obvoznica, mi uporabljamo navadno kar na glavni cesti.
Ko na PIC-u napišejo na gorenjski avtocesti proti Kranju, na dolenjski avtocesti proti Novemu mestu, na podravski avtocesti proti Ptuju, na pomurski avtocesti proti Murski Soboti, … pišemo končne destinacije! Torej proti Avstriji/Karavankam, proti Hrvaški/Obrežju/Gruškovju, proti Madžarski…

SESTAVA PROMETNE INFORMACIJE:

1.	Formulacija
Cesta in smer + razlog + posledica in odsek

2.	Formulacija
Razlog + cesta in smer + posledica in odsek

NUJNE PROMETNE INFORMACIJE
Nujne prometne informacije se najpogosteje nanašajo na zaprto avtocesto; nesrečo na avtocesti, glavni in regionalni cesti; daljši zastoji (neglede na vzrok); pokvarjena vozila, ko je zaprt vsaj en prometni pas; Pešci, živali in predmeti na vozišču ter seveda voznik v napačni smeri. Živali in predmete lahko po dogovoru izločimo.
Zelo pomembne nujne informacije objavljamo na 15 - 20 minut; Se pravi vsaj 2x med enimi in drugimi novicami, ki so ob pol. V pomembne nujne štejemo zaprte avtoceste in daljše zastoje. Tem informacijam je potrebno še bolj slediti in jih posodabljati.
HIERARHIJA DOGODKOV
Voznik v napačno smer 
Zaprta avtocesta
Nesreča z zastojem na avtocesti
Zastoji zaradi del na avtocesti (ob krajših zastojih se pogosto dogajajo naleti)
Zaradi nesreče zaprta glavna ali regionalna cesta
Nesreče na avtocestah in drugih cestah
Pokvarjena vozila, ko je zaprt vsaj en prometni pas
Žival, ki je zašla na vozišče
Predmet/razsut tovor na avtocesti
Dela na avtocesti, kjer je večja nevarnost naleta (zaprt prometni pas, pred predori, v predorih, …)
Zastoj pred Karavankami, napovedi (glej poglavje napovedi)
"""

# """ + pravila_za_formulacijo + """

prompt_example = """Si zaposlen na RTVSlo in odgovoren za prometna poročila. Poročila so opremljena z informacijo o datumu in vsebujejo informacije o prometu brez podnaslovov, strnjena v enotno, jasno razumljivo prometno poročilo. Iz danih vsebinskih podatkov sestavi prometno poročilo, podatke brez vsebinskih naslovov poveži v smiselno enoto.

Poročilo naj ima obliko RTVSlo prometnih poročil, ki so zelo strukturirana in imajo obliko. Strogo se drži te oblike, nadomesti podatke s podanimi.

OBLIKA:
"Prometne informacije     [datum]   [ura]        x. program

Podatki o prometu.

[podatki o nesrečah]

[podatki o zastojih]

[podatki o ovirah]

[podatki o delu na cesti, če so]

[podatki o opozorilih, če so]

[podatki o vremenu, če vreme vpliva na promet]
"

Vhodni podatki, s katerimi ustvari poročilo:
Timestamp: 2024-05-25 16:30:00
Nesreče na cesti:
Zastoji:
Ovire: Na štajerski avtocesti je pri počivališču Tepanje proti Mariboru okvara tovornega vozila na pospeševalnem pasu. Okvara tovornega vozila na primorski avtocesti med predorom Kastelec in Kozino proti Ljubljani.
Delo na cesti: Na gorenjski avtocesti med Kosezami in Šentvidom promet poteka po dveh zoženih pasovih v obe smeri. Hkrati so zaprti:- uvoz Šentvid, s Celovške ceste v predor proti Kosezam;- uvoz Podutik, s Podutiške ceste proti Kranju;- na razcepu Koseze krak iz smeri severne ljubljanske obvoznice proti Gorenjski, obvoz po Celovški cesti ali preko priključka Brdo.Cesta Lesce - Kamna Gorica - Lipnica bo v Kamni Gorici zaprta do nedelje, 26. maja, do 19. ure.Več o delovnih zaporah v prometni napovedi.
Opozorila:
Vreme:
Splosno: Na cesti Ormož - Središče ob Dravi bodo v nedeljo, 26. 5., med 9:30 in 16:30 kratkotrajne 30 minutne popolne zapore."

Podatke iz vsebinskih podatkov poveži v smiselno enoto. V poročilo ne vključi podnaslovov ali hiperpovezave, ki vodi do več informacij, ali na spletno stran promet.si. Najbolj pomembne so informacije o prometnih nesrečah, potem informacije o zastojih in ovirah. Če ni pomembnih podatkov o vremenu, opozoril in splošnih informacij, jih ne vključi. """

response: ChatResponse = chat(model='llama3:8b', messages=[
  {
    'role': 'user',
    'content': prompt_example,
  },
])
print(response['message']['content'])
print(response.message.content)

prometno_porocilo_initial = response['message']['content']

timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
file_name = "prometno_porocilo_initial_" + timestamp + ".txt"

with open(file_name, "w", encoding="utf-8") as f:
    f.write(prometno_porocilo_initial)

next_prompt = "Popravi slovnične napake in stil poročila, da bo bolj tekoče, zgoščeno in razumljivo. Ohrani jezik besedila - naj bo v slovenščini. Ohrani informacije in strukturo poročila. Odstrani naslove, jedro besedila naj bo povezano brez naslov. Prva vrstica naj ostane enaka. " + prometno_porocilo_initial

response: ChatResponse = chat(model='llama3:8b', messages=[
  {
    'role': 'user',
    'content': next_prompt,
  },
])
print(response['message']['content'])
print(response.message.content)

prometno_porocilo = response['message']['content']

timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
file_name = "prometno_porocilo_" + timestamp + ".txt"

with open(file_name, "w", encoding="utf-8") as f:
    f.write(prometno_porocilo)
