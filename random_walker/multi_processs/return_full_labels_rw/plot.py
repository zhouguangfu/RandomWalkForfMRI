__author__ = 'zhouguangfu'

import numpy as np
import matplotlib.pyplot as plt
import datetime

from configs import *

SESSION_NUMBERS = 7

BACKGROUND_MAKRERS_THR = [-3, -2, -1, 0, 1, 2, 3] #len 7 default - (-1)
OBJECT_MARKERS_NUM = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] #len 10 default - 30
ATLAS_SELECTED = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150] #len 12 default - 30

all_test_means = np.zeros((len(OBJECT_MARKERS_NUM), len(ROI), len(BACKGROUND_MAKRERS_THR))) #session. roi, test
all_test_stds = np.zeros((len(OBJECT_MARKERS_NUM), len(ROI), len(BACKGROUND_MAKRERS_THR))) #session. roi, test

# #bk_neg_3
all_means =  [[0.063365660029210805, 0.30042256314984794, 0.53064378455260397, 0.5961836111340223], [0.14518202822983489, 0.41815534355463946, 0.46726498039306275, 0.63961230831965898], [0.17021773960730413, 0.54034390519974607, 0.47821384398799804, 0.64024338860622965], [0.25379982521012145, 0.51942939988241554, 0.50207313852105429, 0.65594163332663535], [0.29653766380152141, 0.48489183496504795, 0.52151229430092039, 0.60817314732926742], [0.34390088180222683, 0.4454929536009703, 0.49302066599085359, 0.56894360362737062], [0.37983449331249702, 0.43486893143484096, 0.42159427884518702, 0.56663300657740956], [0.43472528871085026, 0.4411940725819895, 0.41480694616243047, 0.59132574097932278], [0.49398887608659869, 0.44967952246591447, 0.4188780909367914, 0.5617360453219602], [0.52261726206247527, 0.48431214883997198, 0.45001612937367336, 0.5259165294745809]]
all_stds =  [[0.13170185458104192, 0.23624176390721482, 0.15040930560378701, 0.16372289608731996], [0.1343571987557502, 0.24598924578692202, 0.18961094456218192, 0.14118482403543217], [0.11846336905219647, 0.14565040677588384, 0.23143053867205135, 0.12708114600952988], [0.16178246900845311, 0.17839807431551, 0.25370481742931622, 0.10103197608815699], [0.21025726034863484, 0.18178429981682767, 0.25962963325390243, 0.098454561847110364], [0.23580225230242224, 0.16842704581530929, 0.2288044597573681, 0.12432684413961127], [0.24214087946799268, 0.18025392391819181, 0.21234336679299939, 0.13213647980356158], [0.25183590047168902, 0.20898155397859283, 0.21942649944266093, 0.13826782814899918], [0.24471452404441571, 0.22228113766765156, 0.21715205880551339, 0.1375489992076665], [0.24069097660813865, 0.23586681281276056, 0.1832797066027097, 0.14340926070859997]]
all_test_means[..., 0] = all_means
all_test_stds[..., 0] = all_stds

# #bk_neg_2
all_means = [[0.017058137597867732, 0.23812725781285093, 0.50014150867498319, 0.53903297521319293], [0.062077160239014037, 0.37111767209557167, 0.54641123376553102, 0.65571793149540736], [0.1198115523696935, 0.3970286519254127, 0.58646414885631359, 0.64516635000105083], [0.16853073595977916, 0.411810515200698, 0.6165561087967466, 0.63867869159135038], [0.1842417067272277, 0.41692439224612687, 0.64484294836666634, 0.65448177554344633], [0.18113053783366426, 0.42182657742631691, 0.65583782597892637, 0.65292444581143727], [0.20973010038194945, 0.41201646264914044, 0.63402023139346675, 0.63060774407515496], [0.22033283275355586, 0.40109363562308592, 0.61922542961265692, 0.65142990067626338], [0.24340948091468903, 0.39661739627544212, 0.60235610443677601, 0.63806852912973766], [0.27249619783154677, 0.39677217528880437, 0.58001913601865662, 0.63798371972962453]]
all_stds = [[0.042680208137443804, 0.23752541375193517, 0.17527539122575148, 0.19596078368953943], [0.13104500376244405, 0.26525898509819107, 0.14333875478670605, 0.14698837301022144], [0.16059373881373551, 0.26628942359753222, 0.12371647365754955, 0.14226851007397429], [0.18131784077393315, 0.27281756478218194, 0.10887067322156896, 0.13193888733569897], [0.19758663400338422, 0.27129808587148818, 0.079711091739814582, 0.12891145526570472], [0.1864843507234113, 0.27653730581630637, 0.071244477561930808, 0.10873264546571192], [0.18870737652012559, 0.27382504401956037, 0.08629351367768584, 0.092857863257614512], [0.18860702564420345, 0.28716280378336917, 0.10372511968233637, 0.083825522474994751], [0.19394728080930027, 0.29065312943391725, 0.12428064129607329, 0.088889513378450399], [0.20942426387236193, 0.2866139019274308, 0.13260933080720388, 0.096130714717528973]]
all_test_means[..., 1] = all_means
all_test_stds[..., 1] = all_stds

# #bk_neg_1
all_means = [[0.0, 0.1712667150696901, 0.45790150012380337, 0.50994402613374401], [0.041234824124251078, 0.34313388679475099, 0.51839638484793138, 0.64238743059356374], [0.066357219762305034, 0.36585804902187435, 0.54919785127227372, 0.65366083465906455], [0.11733746653508981, 0.39773201189536567, 0.58869021814082712, 0.65003775498784877], [0.15820505974823856, 0.40814174606236731, 0.61654061515569747, 0.65483634947626013], [0.16771792957154383, 0.41619123414595788, 0.63368876349041925, 0.67873287190233855], [0.19033967307200278, 0.42006100119152961, 0.6427717956189225, 0.65299494296848992], [0.1995139988578026, 0.43551107210826601, 0.65221495265151275, 0.64749208530169988], [0.1994429757666952, 0.43618108362039221, 0.65106545856039089, 0.63854290757366827], [0.20944859973518545, 0.43388962379009544, 0.6468920115098683, 0.62828631424004422]]
all_stds = [[0.0, 0.22951297494692932, 0.18574918499368073, 0.20068046402222323], [0.088635676454857204, 0.27307786429110126, 0.15850898959250692, 0.15981318410163012], [0.11246060639652353, 0.26621945118789131, 0.13232440816274374, 0.16842014929752364], [0.15313599039340325, 0.27094387564876815, 0.1261224688474003, 0.1429047296515891], [0.18728815537927462, 0.27683465713320193, 0.11010532333384575, 0.14693693236620561], [0.18962302006932613, 0.27967538160453376, 0.097527513002323596, 0.12388418792386978], [0.20960601588934227, 0.27057896486824573, 0.083024496934958725, 0.10819643050800444], [0.20766719645439391, 0.28033950872782154, 0.074426885154385253, 0.10281709705565817], [0.20859533745213391, 0.28448620343699799, 0.062636343779111009, 0.098000227040537916], [0.20695625563063128, 0.28692386467017511, 0.064059582939024987, 0.10073356465315067]]
all_test_means[..., 2] = all_means
all_test_stds[..., 2] = all_stds

# #bk_neg_0
all_means = [[0.0, 0.15400231851128371, 0.43163744558582834, 0.48943939446173801], [0.017731768031888077, 0.30000528096679513, 0.51205031741386187, 0.627427583502566], [0.035544871837278273, 0.34134283924059822, 0.54179342489395932, 0.63222611609425206], [0.078985724082328887, 0.36588763977059946, 0.57400519167495445, 0.62678905837497034], [0.11601774777448395, 0.36788113442236542, 0.59124530103984163, 0.62354062313552372], [0.13189527918585275, 0.37288964804133684, 0.60139052686878391, 0.63923052984820272], [0.1564214846387085, 0.38247879465332635, 0.63169380180837642, 0.61033140600284075], [0.15938031617253101, 0.38988366749731185, 0.6343101070923608, 0.61392843936297969], [0.15742448820260013, 0.39022394458566961, 0.64315676230550434, 0.59791665861986676], [0.15450857023696829, 0.39062821605234127, 0.63039494410403529, 0.58896398213088053]]
all_stds = [[0.0, 0.21370131407982035, 0.19393397932309689, 0.19437030472025812], [0.045676246183634119, 0.27982351218050022, 0.16512781319692643, 0.16263809387171371], [0.078958943757671582, 0.27305482058516489, 0.12903547933642856, 0.17792533060193516], [0.12016918464985331, 0.2654025401518294, 0.13401791904319574, 0.16664410426731083], [0.1668491824690928, 0.26355745491395305, 0.13312121895197798, 0.17705462413978393], [0.17737226917654395, 0.2668670374268442, 0.11978728540943123, 0.16481112750386914], [0.20674946083890461, 0.27394507894150427, 0.098015878000131382, 0.14575994949162027], [0.20932990561045028, 0.27713195455921574, 0.087199186982784074, 0.1347751117757108], [0.20183130098703678, 0.27862457368643601, 0.076842708184297032, 0.12231922555860227], [0.20129048872491631, 0.28309707093517211, 0.086696827320159264, 0.12021716927298413]]
all_test_means[..., 3] = all_means
all_test_stds[..., 3] = all_stds

# #bk_1
all_means = [[0.0, 0.10110944725554323, 0.41007735234449622, 0.42189531536440972], [0.013975155279503108, 0.24588366261278954, 0.50442456923244927, 0.59405404896505198], [0.023405304150645768, 0.26567907220534981, 0.51552772082330045, 0.616664435240452], [0.040293307660020067, 0.28960017406916255, 0.5448033431121001, 0.59910461677027749], [0.083762534219767357, 0.31246260335816406, 0.57214712298606063, 0.5998769579685318], [0.1145949634049539, 0.30630567669318204, 0.57788191016004631, 0.59682875970448124], [0.12480133736082928, 0.30058836070395789, 0.59324683766773822, 0.57338363082212018], [0.12545026278083901, 0.30763211832014009, 0.6015072672707833, 0.55232819857260351], [0.12292633297050677, 0.30945455591304871, 0.6079397384094426, 0.54603071254094848], [0.11898850387830159, 0.30490897328702549, 0.59890964612824837, 0.54108323466076891]]
all_stds = [[0.0, 0.19586665182454632, 0.19893048399210747, 0.20962135439310517], [0.043800932204382581, 0.27291551085635612, 0.1736000487042855, 0.16649370468846816], [0.061666896382487527, 0.28819417117407953, 0.14733686546876104, 0.18001460043539713], [0.091377786147637044, 0.28379208794719785, 0.14954450796169302, 0.17899126647392197], [0.1285461001669187, 0.27895356588374964, 0.14969482944138787, 0.19189050479534547], [0.16109040154734192, 0.26874523251650007, 0.13977052467894488, 0.18879233895883049], [0.17172525051909382, 0.27485466261295088, 0.13831838753277059, 0.1823749285704882], [0.17231474682426709, 0.27693349293892727, 0.12434375952010483, 0.18101255447112216], [0.16943767489057671, 0.27374077711025541, 0.12023865134613074, 0.185756107720323], [0.16329606473334249, 0.27342114460575401, 0.12509891898352204, 0.18256586563047453]]
all_test_means[..., 4] = all_means
all_test_stds[..., 4] = all_stds

# #bk_2
all_means =  [[0.0, 0.091755746293885096, 0.35544608028657676, 0.33351665105618877], [0.012987012987012986, 0.18262356986043848, 0.45055775726252489, 0.45554532547500903], [0.012030075187969926, 0.21863028199269538, 0.46981174473019816, 0.49625790542315051], [0.026374401612496851, 0.21964533609672626, 0.49756983143192113, 0.49594909405555043], [0.041410018552875703, 0.22229575462493067, 0.50797512844282811, 0.49949651863356792], [0.074888387276477661, 0.21952664988035619, 0.50692853432611595, 0.50013709737418788], [0.091750093148741593, 0.2243349027459674, 0.51899980971589499, 0.4764405880809362], [0.08902507102381782, 0.22785305371873374, 0.52352707592590109, 0.46195274633250383], [0.083833398636115095, 0.22810392494677925, 0.52964655355120727, 0.45420513275488261], [0.0775635293940483, 0.22822207493033178, 0.52499820738999714, 0.4560458575046345]]
all_stds =  [[0.0, 0.17631132501341298, 0.25505081032593585, 0.21726900357897974], [0.058079687727267258, 0.2274953735080697, 0.22106734149801391, 0.24924621776157974], [0.043496527618880482, 0.25007248031213192, 0.2068919936100122, 0.25245575698083061], [0.070160610504004572, 0.25105175316603456, 0.18168440903431562, 0.24676242318670522], [0.080330792109631829, 0.25236129028678939, 0.19486264979370535, 0.26214176129227262], [0.12196251317915428, 0.25214128724663259, 0.19806430399323141, 0.26139337227674875], [0.14854321151734634, 0.25642220492154905, 0.20683439837910914, 0.25344998053479323], [0.14574132867367692, 0.26146407784820297, 0.20448701431762448, 0.25001031198320772], [0.13840492066276577, 0.26334734489021527, 0.19833456263088767, 0.24917639790765195], [0.12596683958476562, 0.26636071826905977, 0.20346078293718159, 0.24652689954692464]]
all_test_means[..., 5] = all_means
all_test_stds[..., 5] = all_stds

# #bk_3
all_means = [[0.0, 0.070526104773648401, 0.32501105671384201, 0.2341043201761647], [0.0, 0.10256245432896736, 0.40345169214770088, 0.32298304777230596], [0.0095238095238095247, 0.09548632198078924, 0.42351005227011235, 0.38154273008998874], [0.0070546737213403876, 0.10878227155966803, 0.43808083957000016, 0.41742182403910233], [0.013664596273291927, 0.11465710816637992, 0.46015758893582465, 0.43134525017123482], [0.025888704932035555, 0.11810954407362734, 0.46076995572714691, 0.43691822695594723], [0.024163832199546483, 0.12003546884165187, 0.47927840605757843, 0.41325008598389862], [0.027131061029366115, 0.11803357113929858, 0.48273517133612032, 0.41633109486494696], [0.029474273138331782, 0.11796163631620982, 0.49408411358593018, 0.41512855090682782], [0.026234567901234566, 0.11758460285323979, 0.49356722029005257, 0.41340958419833163]]
all_stds = [[0.0, 0.12396198919308289, 0.25652622320634755, 0.25597369480053339], [0.0, 0.20250395298191307, 0.27096421392775399, 0.25873125766028393], [0.042591770999996011, 0.17281709309967366, 0.2601266686451611, 0.27437419953741576], [0.031549459999997045, 0.19044738273843081, 0.24151696100323986, 0.28199090885305422], [0.045586282217881198, 0.19600711619845365, 0.22128865157465016, 0.29884690054510626], [0.069075447529301642, 0.20363880256526951, 0.21900290464148583, 0.30064799605857789], [0.066972796073454147, 0.20770649463894039, 0.22741290365008229, 0.28021438958412309], [0.078387718806752091, 0.20429029397235229, 0.2197274947876062, 0.28427078164124048], [0.082155096098773514, 0.20557549787489859, 0.21364408593122935, 0.28121969152407711], [0.067670362675456203, 0.20414614207146306, 0.22009835445001139, 0.272852453453999]]
all_test_means[..., 6] = all_means
all_test_stds[..., 6] = all_stds

#atlas selected
# all_means = [[0.16281425353749657, 0.38568491041149433, 0.58713985710897454, 0.66943066421050634], [0.16038919154636622, 0.40268114120228221, 0.63084501542143301, 0.65006687989529954], [0.16771792957154383, 0.41619123414595788, 0.63368876349041925, 0.67873287190233855], [0.1235953492324383, 0.37332501689948627, 0.63978025158074714, 0.62939463903205495], [0.077320299906434214, 0.31820497529849084, 0.63612235549137153, 0.64075268794438944], [0.079636493751482879, 0.27332071248729567, 0.63473702391017028, 0.64143948653690563], [0.090556495295070843, 0.27308077696479099, 0.63583023547835893, 0.66657873711442039], [0.094872684746612007, 0.28692228766758576, 0.63288258128182917, 0.65810250387782099], [0.096544345940948545, 0.28398712960856942, 0.63466347162112624, 0.65773676114879287], [0.1275596681354256, 0.28084988551986712, 0.63350009701024013, 0.66954318612714214]]
# all_stds = [[0.15370705240562985, 0.19075451533904045, 0.11718139735144648, 0.13477686444425246], [0.16586478420618811, 0.2317763820421937, 0.10981817642811229, 0.12276067111552308], [0.18962302006932613, 0.27967538160453376, 0.097527513002323596, 0.12388418792386978], [0.13925202359827024, 0.26777090010102322, 0.093256251657012776, 0.16009933129852016], [0.12507905319942997, 0.2682455238989519, 0.090990903039540716, 0.14361555724468267], [0.12374083549526017, 0.29203253744482049, 0.091684853105704955, 0.15432171979606665], [0.13284973740883876, 0.27959164260559943, 0.090464478766109607, 0.15284235334711813], [0.13190463660277421, 0.2868758265142502, 0.09379978996464447, 0.15889972656614332], [0.14870631156919881, 0.27709119200762028, 0.091331277916119974, 0.15512527750310351], [0.15617274985303981, 0.27346527496354339, 0.092157821182580771, 0.15330867298826506]]

#atlas selected (<0 is 0)
all_means = [[0.20580098834110752, 0.54254616219348406, 0.61700775105130745, 0.67368000863705735], [0.26503303982558918, 0.45294084116840061, 0.6353274290239721, 0.66279650210776853], [0.23952379361869958, 0.43314500029992253, 0.63304734858374678, 0.67470830732519627], [0.21468522327630629, 0.46779466227753036, 0.6286292388518393, 0.66177255467768759], [0.17856340114672625, 0.46115145650021039, 0.63432903259872397, 0.6464746325086993], [0.17852992061449199, 0.42771951122601998, 0.63174214097615189, 0.62917687654688137], [0.15885129448318938, 0.39285257117679828, 0.62562158397884204, 0.63729842539621606], [0.14486335820817359, 0.362038892565957, 0.62899260401109336, 0.64284801377557044], [0.12967157236369187, 0.31657390855557244, 0.62402569187081425, 0.6625771335455849], [0.14954096972088954, 0.33608882785855815, 0.62569635308221427, 0.67251029990471578]]
all_stds = [[0.20766918194574199, 0.15038692445389676, 0.10278950535487745, 0.11217115928910211], [0.21288604269343156, 0.27261937194968167, 0.091263148642304281, 0.098074329716187889], [0.20488490027064191, 0.28518500940093117, 0.098952177965151239, 0.099577194357617041], [0.18488279524851711, 0.26208603936801889, 0.089148175817974393, 0.091878002926190222], [0.17385618528845906, 0.25591490851685433, 0.090508469868501265, 0.1012941528851586], [0.16316486582312406, 0.28342102337175062, 0.090168416407852503, 0.12865131456563278], [0.16254442134121314, 0.27761886613244724, 0.085932710950343563, 0.15277607319238909], [0.1551738382878888, 0.23236656497407612, 0.09013972818698314, 0.16239446273031058], [0.14889713158144671, 0.26311749293277503, 0.093933418709371469, 0.1596538998872209], [0.16927450263110558, 0.25903884411793282, 0.092169690886047456, 0.1577668026063343]]


if __name__ == "__main__":
    starttime = datetime.datetime.now()


    # all_test_means[:, 0, :] = 0
    # all_test_means[:, 2, :] = 0
    # all_test_stds[:, 0, :] = 0
    # all_test_stds[:, 2, :] = 0
    # roi_means = np.average(all_test_means, axis=1)
    # roi_stds = np.average(all_test_stds, axis=1)
    # roi_means = all_test_means[:, 3, :]
    # roi_stds = all_test_stds[:, 3, :]
    # roi_stds[roi_stds == 0] = 100

    # roi_means = roi_means / roi_stds

    # print roi_means.shape
    #
    # import prettyplotlib as ppl
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # with ppl.pretty:
    #     fig, ax = plt.subplots(1)
    # green_purple = ppl.brewer2mpl.get_map('PRGn', 'diverging', 11).mpl_colormap
    #
    # ax.set_xlabel("Background Threshold",fontsize=12)
    # ax.set_ylabel("Foreground Threshold",fontsize=12)
    # ax.set_title("Left Brain Avearge",fontsize=15)
    # # ax.set_title("l_OFA Average",fontsize=15)
    # ppl.pcolormesh(fig, ax, roi_means, xticklabels=BACKGROUND_MAKRERS_THR, yticklabels=OBJECT_MARKERS_NUM, cmap=green_purple)
    # plt.show()
    # fig.savefig(RW_RESULT_DATA_DIR + 'temp.png')

    all_means = np.array(all_means)
    all_stds = np.array(all_stds)

    all_means = all_means / all_stds

    plt.plot(OBJECT_MARKERS_NUM, all_means[:, 0].tolist(), '--rx', alpha=0.8, label='r_OFA')
    plt.plot(OBJECT_MARKERS_NUM, all_means[:, 1].tolist(), '--g*', alpha=0.8, label='l_OFA')
    plt.plot(OBJECT_MARKERS_NUM, all_means[:, 2].tolist(), '--b^', alpha=0.8, label='r_pFus')
    plt.plot(OBJECT_MARKERS_NUM, all_means[:, 3].tolist(), '--yo', alpha=0.8, label='l_pFus')


    legend = plt.legend(bbox_to_anchor=(1., 0.5), loc='center left', shadow=True)
    plt.savefig(RW_RESULT_DATA_DIR + 'temp.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()



    endtime = datetime.datetime.now()













