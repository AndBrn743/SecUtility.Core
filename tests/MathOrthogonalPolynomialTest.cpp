//
// Created by Andy on 5/16/2026.
//

#include <SecUtility/Math/OrthogonalPolynomial.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iomanip>
#include <iostream>


TEST_CASE("Rys roots and weights")
{
	using namespace SecUtility::Math;

	SECTION("Shifted Legendre")
	{
		const auto auxSize = 100;
		QuadratureGrid grid = GenerateFejerQuadratureGrid01<long double>(auxSize);

		const auto degree = 50;
		const auto jacobiRules = ConstructOrthogonalPolynomialRecurrence(grid, degree);

		REQUIRE(jacobiRules.Alphas.size() == degree);
		REQUIRE(jacobiRules.Gammas.size() == degree);

		for (const auto alpha : jacobiRules.Alphas)
		{
			CHECK(alpha == Catch::Approx(0.5).margin(1e-16));
		}

		REQUIRE(jacobiRules.Gammas[0] == 0);  // must be exactly zero even as a floating point number
		for (Eigen::Index i = 1; i < degree; ++i)
		{
			CHECK(jacobiRules.Gammas[i] == Catch::Approx(i / (2 * Sqrt(4 * i * i - 1))).margin(1e-16));
		}

		const auto& [roots, weights] = CalculateOrthogonalPolynomialNodesAndWeightsFrom(jacobiRules, 1.L);
		for (Eigen::Index i = 0; i < degree * 2; ++i)
		{
			CHECK(roots.cwisePow(i).dot(weights) == Catch::Approx(1.0l / (i + 1)).margin(1e-14));
		}
	}

	SECTION("Exp 1")
	{
		const auto auxSize = 100;

		QuadratureGrid grid = GenerateFejerQuadratureGrid01<long double>(auxSize);
		for (int i = 0; i < auxSize; ++i)
		{
			grid.Weight(i) *= Exp(-grid.Node(i));
		}
		std::cout << "Aux Roots: " << grid.Nodes().transpose() << std::endl;
		std::cout << "Aux Weights: " << grid.Weights().transpose() << std::endl;

		const auto degree = 50;
		const auto jacobiRules = ConstructOrthogonalPolynomialRecurrence(grid, degree);

		std::cout << "Alphas: " << jacobiRules.Alphas.transpose() << std::endl;
		std::cout << "Gammas: " << jacobiRules.Gammas.transpose() << std::endl;

		Eigen::MatrixX<long double> jacobian = Eigen::MatrixX<long double>::Zero(degree, degree);
		jacobian.diagonal() = jacobiRules.Alphas;
		jacobian.diagonal(1) = jacobiRules.Gammas.segment(1, degree - 1);
		jacobian.diagonal(-1) = jacobian.diagonal(1);

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<long double>> es(jacobian);

		// Mathematica: Table[NumberForm[N@Integrate[t^n E^(-SetPrecision[1.0,250] t),{t,0,1}],16],{n,0,50 2}]
		const std::vector referenceMoments = {
		        0.6321205588285577,   0.2642411176571153,   0.1606027941427884,   0.1139289412569229,
		        0.0878363238562491,   0.07130217810980316,  0.05993362748737664,  0.05165595124019414,
		        0.0453681687501108,   0.04043407757955496,  0.03646133462410727,  0.03319523969373767,
		        0.03046343515340977,  0.02814521582288474,  0.02615358034894402,  0.02442426406271792,
		        0.0229087838320443,   0.02156988397331076,  0.02037847034815143,  0.01931149544343493,
		        0.01835046769725621,  0.01748038047093801,  0.01668892918919393,  0.01596593018001798,
		        0.0153028831489891,   0.01469263755328523,  0.01412913521397367,  0.01360720960584677,
		        0.01312242779226732,  0.01267096480431004,  0.01224950295785891,  0.0118551505221838,
		        0.01148537553843909,  0.01113795159704749,  0.01081091312817218,  0.01050251831458408,
		        0.01021121815358472,  0.00993563051119255,  0.0096745182538747,   0.00942677072967099,
		        0.00919138801539721,  0.00896746745984341,  0.00875419214198098,  0.00855082093373972,
		        0.00835667991310521,  0.00817115491829197,  0.007993685069988094, 0.00782375711799807,
		        0.007660900492465045, 0.00750468295934485,  0.00735470679580019,  0.00721060541436731,
		        0.007072040375657823, 0.006938698738422277, 0.006810290703360586, 0.006686547513389912,
		        0.006567219578392738, 0.006452074796943795, 0.0063408970512978,   0.006233484855127918,
		        0.006129650136232766, 0.006029217138756416, 0.005932021431455491, 0.005837909010253637,
		        0.005746735484790436, 0.005658365339936015, 0.005572671264334772, 0.005489533538987407,
		        0.005408839479701395, 0.005330482927953948, 0.005254363785334027, 0.005180387587273558,
		        0.005108465112253877, 0.005038512023090671, 0.004970448537267357, 0.004904199123609468,
		        0.004839692222877236, 0.004776859990104868, 0.004715638056737411, 0.004655965310813216,
		        0.004597783693614938, 0.004541038011367598, 0.004485675760700689, 0.004431646966714835,
		        0.004378904032603808, 0.004327401599881309, 0.004277096418350276, 0.004227947225031622,
		        0.004179914631340498, 0.004132961017861979, 0.004087050436135752, 0.004042148516911118,
		        0.003998222384380474, 0.003955240575941707, 0.003913172967078121, 0.003871990700979155,
		        0.003831666122556505, 0.00379217271653873,  0.003753485049353237, 0.003715578714528098,
		        0.003678430281367489};

		const Eigen::VectorX<long double> roots = es.eigenvalues();
		const Eigen::VectorX<long double> weights = referenceMoments[0] * es.eigenvectors().row(0).cwiseAbs2();
		std::cout << "Roots: " << std::scientific << std::setprecision(16) << roots.transpose() << std::endl;
		std::cout << "Weights: " << weights.transpose() << std::endl;

		for (Eigen::Index i = 0; i < degree * 2; ++i)
		{
			CHECK(roots.cwisePow(i).dot(weights) == Catch::Approx(referenceMoments[i]).margin(1e-14));
		}
	}

	SECTION("Exp 10")
	{
		const auto auxSize = 100;

		QuadratureGrid grid = GenerateFejerQuadratureGrid<long double>(auxSize);
		grid.Nodes() = (grid.Nodes() + Eigen::VectorX<long double>::Constant(auxSize, 1)) / 2;
		for (int i = 0; i < auxSize; ++i)
		{
			grid.Weight(i) *= Exp(-10 * grid.Node(i)) / 2;
		}
		std::cout << "Aux Roots: " << grid.Nodes().transpose() << std::endl;
		std::cout << "Aux Weights: " << grid.Weights().transpose() << std::endl;

		const auto degree = 50;
		const auto jacobiRules = ConstructOrthogonalPolynomialRecurrence(grid, degree);

		std::cout << "Alphas: " << jacobiRules.Alphas.transpose() << std::endl;
		std::cout << "Gammas: " << jacobiRules.Gammas.transpose() << std::endl;

		Eigen::MatrixX<long double> jacobian = Eigen::MatrixX<long double>::Zero(degree, degree);
		jacobian.diagonal() = jacobiRules.Alphas;
		jacobian.diagonal(1) = jacobiRules.Gammas.segment(1, degree - 1);
		jacobian.diagonal(-1) = jacobian.diagonal(1);

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<long double>> es(jacobian);

		// Mathematica: Table[NumberForm[N@Integrate[t^n E^(-SetPrecision[10,150] t),{t,0,1}],16],{n,0,50 2}]
		const std::vector referenceMoments = {
		        0.0999954600070238,     0.00999500600772613,    0.001994461208568977,   0.0005937983695944445,
		        0.0002329793548615294,  0.0001119496844545162,  0.00006262981769646123, 0.00003930087941127437,
		        0.00002690071055277101, 0.00001967064652124543, 0.00001513065354499694, 0.00001210372592324815,
		        9.9844781316493e-6,     8.4398285948956e-6,     7.275767056605353e-6,   6.373657608659546e-6,
		        5.657859197606788e-6,   5.078367659683057e-6,   4.601068811181015e-6,   4.202037764995445e-6,
		        3.864082553742405e-6,   3.574580386610565e-6,   3.324083874294757e-6,   3.105399934629457e-6,
		        2.912966866862212e-6,   2.742424190907044e-6,   2.590309920109828e-6,   2.45384380804805e-6,
		        2.330769686286056e-6,   2.219239113981076e-6,   2.117724365694742e-6,   2.024952557405213e-6,
		        1.939855207448196e-6,   1.861529208330561e-6,   1.789206332075421e-6,   1.722229186015489e-6,
		        1.660032093407275e-6,   1.602125769358434e-6,   1.548084947313564e-6,   1.497538318274415e-6,
		        1.450160296849174e-6,   1.405664240833128e-6,   1.363796835250652e-6,   1.324333415329318e-6,
		        1.287074051200517e-6,   1.25184025415384e-6,    1.21847219285918e-6,    1.186826330189662e-6,
		        1.156773408661893e-6,   1.12819672619479e-6,    1.100990654725465e-6,   1.075059362851386e-6,
		        1.050315710578722e-6,   1.026680289818742e-6,   1.004080588772724e-6,   9.82450262001498e-7,
		        9.61728490959904e-7,    9.41859422222967e-7,    9.2279167264472e-7,     9.04477892355365e-7,
		        8.86874377883702e-7,    8.69940728842098e-7,    8.5363954257252e-7,     8.37936141958393e-7,
		        8.2279833228523e-7,     8.0819618360551e-7,     7.94101835547881e-7,    7.80489321922317e-7,
		        7.673344128232708e-7,   7.546144722320838e-7,   7.423083293761021e-7,   7.303961623218401e-7,
		        7.18859392468763e-7,    7.076805887734854e-7,   6.96843380675306e-7,    6.863323788163101e-7,
		        6.76133102755472e-7,    6.662319149686493e-7,   6.566159605069793e-7,   6.472731117566514e-7,
		        6.381919178047267e-7,   6.293615579698009e-7,   6.207717991038822e-7,   6.124129563137377e-7,
		        6.042758567869109e-7,   5.963518064402572e-7,   5.88632559137727e-7,    5.8111028824974e-7,
		        5.737775603492259e-7,   5.66627310859625e-7,    5.596528214881404e-7,   5.528476992935919e-7,
		        5.462058572525608e-7,   5.397214962003303e-7,   5.333890880346189e-7,   5.272033600803947e-7,
		        5.211592805233041e-7,   5.152520448275645e-7,   5.094770630616468e-7,   5.038299480618185e-7,
		        4.983065043696996e-7};

		const Eigen::VectorX<long double> roots = es.eigenvalues();
		const Eigen::VectorX<long double> weights = referenceMoments[0] * es.eigenvectors().row(0).cwiseAbs2();
		std::cout << "Roots: " << std::scientific << std::setprecision(16) << roots.transpose() << std::endl;
		std::cout << "Weights: " << weights.transpose() << std::endl;

		for (Eigen::Index i = 0; i < degree * 2; ++i)
		{
			CHECK(roots.cwisePow(i).dot(weights) == Catch::Approx(referenceMoments[i]).margin(1e-14));
		}
	}

	SECTION("Gaussian 1")
	{
		const auto auxSize = 100;

		QuadratureGrid grid = GenerateFejerQuadratureGrid<long double>(auxSize);
		grid.Nodes() = (grid.Nodes() + Eigen::VectorX<long double>::Constant(auxSize, 1)) / 2;
		for (int i = 0; i < auxSize; ++i)
		{
			grid.Weight(i) *= Exp(-grid.Node(i) * grid.Node(i)) / 2;
		}
		std::cout << "Aux Roots: " << grid.Nodes().transpose() << std::endl;
		std::cout << "Aux Weights: " << grid.Weights().transpose() << std::endl;

		const auto degree = 50;
		const auto jacobiRules = ConstructOrthogonalPolynomialRecurrence(grid, degree);

		std::cout << "Alphas: " << jacobiRules.Alphas.transpose() << std::endl;
		std::cout << "Gammas: " << jacobiRules.Gammas.transpose() << std::endl;

		Eigen::MatrixX<long double> jacobian = Eigen::MatrixX<long double>::Zero(degree, degree);
		jacobian.diagonal() = jacobiRules.Alphas;
		jacobian.diagonal(1) = jacobiRules.Gammas.segment(1, degree - 1);
		jacobian.diagonal(-1) = jacobian.diagonal(1);

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<long double>> es(jacobian);

		// Mathematica: Table[NumberForm[N@Integrate[t^n E^(-SetPrecision[1.0,150] t^2),{t,0,1}],16],{n,0,50 2}]
		const std::vector referenceMoments = {
		        0.746824132812427,    0.3160602794142788,   0.1894723458204924,   0.1321205588285577,
		        0.1002687981450174,   0.0803013970713942,   0.06673227477682225,  0.05696447062846143,
		        0.04962324113315674,  0.04391816192812454,  0.03936486451348415,  0.03565108905490158,
		        0.03256703423844173,  0.02996681374368832,  0.02774600196415005,  0.02582797562009707,
		        0.02415529414540417,  0.0226840843750554,   0.0213802796502143,   0.02021703878977748,
		        0.01917293609131463,  0.01823066731205364,  0.01737610837308246,  0.01659761984686884,
		        0.01588552570472715,  0.01523171757670489,  0.0146293507233682,   0.01407260791144237,
		        0.01355651417974959,  0.01307679017447201,  0.01262973502064794,  0.01221213203135896,
		        0.01182117223432188,  0.01145439191602215,  0.01110962128058987,  0.01078494198665538,
		        0.01047865182460152,  0.01018923517407572,  0.00991533816940703,  0.00965574772171746,
		        0.00940937371771592,  0.0091752338486281,   0.00895244062745517,  0.008740190235469,
		        0.00853775290456507,  0.00834446459459696,  0.00815971976699283,  0.007982965090008988,
		        0.007813693938610239, 0.007651441574494551, 0.007495780910229656, 0.007346318776642614,
		        0.007202692625135065, 0.007064567606986836, 0.006931633980358082, 0.006803604802923386,
		        0.006680213874126083, 0.006561213896133662, 0.006446374826872201, 0.006335482402155019,
		        0.006228336807008763, 0.006124751478929453, 0.006024552028046135, 0.005927575261091897,
		        0.005833668297732087, 0.005742687769219543, 0.005654499090571688, 0.005568975798523743,
		        0.005485998948430409, 0.005405456564086092, 0.005327243135127969, 0.005251259157292042,
		        0.00517741071132173,  0.005105609076792362, 0.005035770377521946, 0.004967815255596276,
		        0.00490166857135178,  0.00483725912693735,  0.004774519411322359, 0.004713385364835494,
		        0.004653796161512043, 0.004595694007698606, 0.004539023955516597, 0.004483733729921706,
		        0.004429773568217616, 0.004377096070990489, 0.004325656063527548, 0.004275410466869858,
		        0.004226318177727169, 0.004178339956552604, 0.004131438323137838, 0.004085577459145982,
		        0.004040723117050443, 0.003996842534994047, 0.003953904357124403, 0.003911878558999035,
		        0.003870736377687992, 0.003830450246232522, 0.003790993732146453, 0.003752341479672425,
		        0.003714469155528287};

		const Eigen::VectorX<long double> roots = es.eigenvalues();
		const Eigen::VectorX<long double> weights = referenceMoments[0] * es.eigenvectors().row(0).cwiseAbs2();
		std::cout << "Roots: " << std::scientific << std::setprecision(16) << roots.transpose() << std::endl;
		std::cout << "Weights: " << weights.transpose() << std::endl;

		for (Eigen::Index i = 0; i < degree * 2; ++i)
		{
			CHECK(roots.cwisePow(i).dot(weights) == Catch::Approx(referenceMoments[i]).margin(1e-14));
		}
	}

	SECTION("Gaussian 10")
	{
		const auto auxSize = 100;

		QuadratureGrid grid = GenerateFejerQuadratureGrid<long double>(auxSize);
		grid.Nodes() = (grid.Nodes() + Eigen::VectorX<long double>::Constant(auxSize, 1)) / 2;
		for (int i = 0; i < auxSize; ++i)
		{
			grid.Weight(i) *= Exp(-10 * grid.Node(i) * grid.Node(i)) / 2;
		}
		std::cout << "Aux Roots: " << grid.Nodes().transpose() << std::endl;
		std::cout << "Aux Weights: " << grid.Weights().transpose() << std::endl;

		const auto degree = 50;
		const auto jacobiRules = ConstructOrthogonalPolynomialRecurrence(grid, degree);

		std::cout << "Alphas: " << jacobiRules.Alphas.transpose() << std::endl;
		std::cout << "Gammas: " << jacobiRules.Gammas.transpose() << std::endl;

		Eigen::MatrixX<long double> jacobian = Eigen::MatrixX<long double>::Zero(degree, degree);
		jacobian.diagonal() = jacobiRules.Alphas;
		jacobian.diagonal(1) = jacobiRules.Gammas.segment(1, degree - 1);
		jacobian.diagonal(-1) = jacobian.diagonal(1);

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<long double>> es(jacobian);

		// Mathematica: Table[NumberForm[N@Integrate[t^n E^(-SetPrecision[10,50] t^2),{t,0,1}],16],{n,0,50 2}]
		const std::vector referenceMoments = {
		        0.2802473905066427,     0.04999773000351188,    0.01401009952884401,    0.004997503003863063,
		        0.002099244932838477,   0.000997230604284488,   0.0005225412367214952,  0.0002968991847972223,
		        0.0001806194363643991,  0.0001164896774307647,  0.00007900874987585535, 0.00005597484222725811,
		        0.0000411848159435962,  0.00003131490884823061, 0.00002450013387521328, 0.00001965043970563718,
		        0.00001610510391828572, 0.00001345035527638551, 0.00001141934184241862, 9.83532326062271e-6,
		        8.57837826217345e-6,    7.565326772498471e-6,   6.737300687157876e-6,   6.051862961624076e-6,
		        5.477899302107314e-6,   4.992239065824648e-6,   4.577377639509901e-6,   4.2199142974478e-6,
		        3.909463325214123e-6,   3.637883528302677e-6,   3.398725333436236e-6,   3.186828804329773e-6,
		        2.998027778701922e-6,   2.828929598803394e-6,   2.676749346733929e-6,   2.539183829841528e-6,
		        2.414314868660133e-6,   2.300534405590508e-6,   2.196486018897004e-6,   2.101018882497722e-6,
		        2.013151248724915e-6,   1.932041276871202e-6,   1.856963571761834e-6,   1.787290193305282e-6,
		        1.722475191163702e-6,   1.662041937147379e-6,   1.605572691994086e-6,   1.552699967314728e-6,
		        1.503099338061859e-6,   1.456483433431106e-6,   1.412596890127313e-6,   1.371212095453522e-6,
		        1.332125581700406e-6,   1.295154960054914e-6,   1.260136303381834e-6,   1.226921904024025e-6,
		        1.1953783461758e-6,     1.165384843143028e-6,   1.136831798476788e-6,   1.109619556990538e-6,
		        1.083657317382281e-6,   1.058862182847371e-6,   1.035158329891714e-6,   1.012476278702606e-6,
		        9.90752251034657e-7,    9.69927603724098e-7,    9.49948327738391e-7,    9.3076460416528e-7,
		        9.12330409799368e-7,    8.94603166037711e-7,    8.77543425683578e-7,    8.61114593007745e-7,
		        8.4528267305246e-7,     8.30016046703638e-7,    8.15285268517236e-7,    8.01062884679217e-7,
		        7.873232688153936e-7,   7.74042473656782e-7,    7.611980968150228e-7,   7.487691591372075e-7,
		        7.367359942950977e-7,   7.25080148424587e-7,    7.137842887709033e-7,   7.028321204165639e-7,
		        6.922083102750058e-7,   6.818984176253261e-7,   6.718888305445317e-7,   6.621667076646593e-7,
		        6.527199247444702e-7,   6.435370256002583e-7,   6.346071769886491e-7,   6.259201270769202e-7,
		        6.174661671741112e-7,   6.092360964295901e-7,   6.012211892353745e-7,   5.934131650948311e-7,
		        5.858041607437865e-7,   5.783867043309464e-7,   5.711536914831221e-7,   5.640983630973951e-7,
		        5.572142847172122e-7};

		const Eigen::VectorX<long double> roots = es.eigenvalues();
		const Eigen::VectorX<long double> weights = referenceMoments[0] * es.eigenvectors().row(0).cwiseAbs2();
		std::cout << "Roots: " << std::scientific << std::setprecision(16) << roots.transpose() << std::endl;
		std::cout << "Weights: " << weights.transpose() << std::endl;

		for (Eigen::Index i = 0; i < degree * 2; ++i)
		{
			CHECK(roots.cwisePow(i).dot(weights) == Catch::Approx(referenceMoments[i]).margin(1e-14));
		}
	}

	SECTION("Exp 30")
	{
		const auto auxSize = 100;

		QuadratureGrid grid = GenerateFejerQuadratureGrid<long double>(auxSize);
		grid.Nodes() = (grid.Nodes() + Eigen::VectorX<long double>::Constant(auxSize, 1)) / 2;
		for (int i = 0; i < auxSize; ++i)
		{
			grid.Weight(i) *= Exp(-30 * grid.Node(i)) / 2;
		}
		std::cout << "Aux Roots: " << grid.Nodes().transpose() << std::endl;
		std::cout << "Aux Weights: " << grid.Weights().transpose() << std::endl;

		const auto degree = 50;
		const auto jacobiRules = ConstructOrthogonalPolynomialRecurrence(grid, degree);

		std::cout << "Alphas: " << jacobiRules.Alphas.transpose() << std::endl;
		std::cout << "Gammas: " << jacobiRules.Gammas.transpose() << std::endl;

		Eigen::MatrixX<long double> jacobian = Eigen::MatrixX<long double>::Zero(degree, degree);
		jacobian.diagonal() = jacobiRules.Alphas;
		jacobian.diagonal(1) = jacobiRules.Gammas.segment(1, degree - 1);
		jacobian.diagonal(-1) = jacobian.diagonal(1);

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<long double>> es(jacobian);

		// Mathematica: Table[NumberForm[N@Integrate[t^n E^(-SetPrecision[30,50] t^2),{t,0,1}],16],{n,0,50 2}]
		const std::vector referenceMoments = {
		        0.03333333333333022,   0.001111111111107888,  0.00007407407407073999, 7.407407403954791e-6,
		        9.87654317408098e-7,   1.64609049782142e-7,   3.292180683722074e-8,   7.681751809477182e-9,
		        2.048464029986259e-9,  6.145360897882215e-10, 2.048422440550842e-10,  7.510570361254127e-11,
		        3.003916223736023e-11, 1.301385109519982e-11, 6.070011303436969e-12,  3.031886444062204e-12,
		        1.613886895843562e-12, 9.11416699988405e-13,  5.437308123367631e-13,  3.412436401570031e-13,
		        2.243765524483887e-13, 1.53944379057592e-13,  1.097733369859541e-13,  8.10403506996181e-14,
		        6.171307290341442e-14, 4.830835309656528e-14, 3.874803169407653e-14,  3.175402086838881e-14,
		        2.65178784875495e-14,  2.251474154835113e-14, 1.939553389207107e-14,  1.692284403219338e-14,
		        1.493182597805955e-14, 1.330580091958545e-14, 1.196070005258345e-14,  1.08349424050673e-14,
		        9.8827232298007e-15,   9.0694843271408e-15,   8.36880582476496e-15,   7.760239915914388e-15,
		        7.227778898272459e-15, 6.758756838025636e-15, 6.343051916955833e-15,  5.972500091356636e-15,
		        5.640459144376341e-15, 5.341481060284453e-15, 5.07106330282277e-15,   4.82545818480895e-15,
		        4.601525439414261e-15, 4.396617228096567e-15, 4.208487723880888e-15,  4.035221474317451e-15,
		        3.875176232536854e-15, 3.726937021201718e-15, 3.589278981883034e-15,  3.461137143838836e-15,
		        3.34158167888577e-15,  3.229797533602905e-15, 3.125067575352225e-15,  3.026758575245985e-15,
		        2.934309494211913e-15, 2.847221648617497e-15, 2.765050417529436e-15,  2.687398220531759e-15,
		        2.613908547521027e-15, 2.544260863348833e-15, 2.478166243087375e-15,  2.415363619948413e-15,
		        2.355616548936344e-15, 2.298710406273533e-15, 2.244449958358187e-15,  2.192657245167649e-15,
		        2.1431697321223e-15,   2.095838691884206e-15, 2.050527783700983e-15,  2.007111802972399e-15,
		        1.965475577916686e-15, 1.925512993706103e-15, 1.887126127355808e-15,  1.850224479090238e-15,
		        1.814724287960575e-15, 1.780547921213495e-15, 1.74762332837016e-15,   1.715883552210719e-15,
		        1.685266289909953e-15, 1.655713498464811e-15, 1.627171039319066e-15,  1.599588357745232e-15,
		        1.572918193105957e-15, 1.547116316600947e-15, 1.522141293522783e-15,  1.497954267405715e-15,
		        1.474518763764134e-15, 1.451800511388758e-15, 1.429767279404716e-15,  1.408388728501542e-15,
		        1.387636274924877e-15, 1.367482965977045e-15, 1.347903365911622e-15,  1.328873451228294e-15,
		        1.310370514480923e-15};

		const Eigen::VectorX<long double> roots = es.eigenvalues();
		const Eigen::VectorX<long double> weights = referenceMoments[0] * es.eigenvectors().row(0).cwiseAbs2();
		std::cout << "Roots: " << std::scientific << std::setprecision(16) << roots.transpose() << std::endl;
		std::cout << "Weights: " << weights.transpose() << std::endl;

		for (Eigen::Index i = 0; i < degree * 2; ++i)
		{
			CHECK(roots.cwisePow(i).dot(weights) == Catch::Approx(referenceMoments[i]).margin(1e-14));
		}
	}

	SECTION("Exp 1e-6")
	{
		const auto auxSize = 100;

		QuadratureGrid grid = GenerateFejerQuadratureGrid<long double>(auxSize);
		grid.Nodes() = (grid.Nodes() + Eigen::VectorX<long double>::Constant(auxSize, 1)) / 2;
		for (int i = 0; i < auxSize; ++i)
		{
			grid.Weight(i) *= Exp(-1e-6 * grid.Node(i)) / 2;
		}
		std::cout << "Aux Roots: " << grid.Nodes().transpose() << std::endl;
		std::cout << "Aux Weights: " << grid.Weights().transpose() << std::endl;

		const auto degree = 50;
		const auto jacobiRules = ConstructOrthogonalPolynomialRecurrence(grid, degree);

		std::cout << "Alphas: " << jacobiRules.Alphas.transpose() << std::endl;
		std::cout << "Gammas: " << jacobiRules.Gammas.transpose() << std::endl;

		Eigen::MatrixX<long double> jacobian = Eigen::MatrixX<long double>::Zero(degree, degree);
		jacobian.diagonal() = jacobiRules.Alphas;
		jacobian.diagonal(1) = jacobiRules.Gammas.segment(1, degree - 1);
		jacobian.diagonal(-1) = jacobian.diagonal(1);

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<long double>> es(jacobian);

		// Mathematica: Table[NumberForm[N@Integrate[t^n E^(-SetPrecision[10^-6,1000] t),{t,0,1}],16],{n,0,50 2}]
		const std::vector referenceMoments = {
		        0.999999500000167,   0.4999996666667917,  0.3333330833334333,  0.2499998000000833,  0.1999998333334048,
		        0.1666665238095863,  0.1428570178571984,  0.1249998888889389,  0.1111110111111566,  0.0999999090909508,
		        0.090909007575796,   0.0833332564102921,  0.07692300549453883, 0.07142850476193602, 0.06666660416669608,
		        0.06249994117649837, 0.05882347385623547, 0.05555550292400161, 0.05263152894739223, 0.04999995238097511,
		        0.04761900216452391, 0.04545450197630542, 0.04347821920291855, 0.0416666266666859,  0.03999996153848006,
		        0.03846150142451928, 0.03703700132276856, 0.03571425123154376, 0.03448272528737245, 0.03333330107528444,
		        0.03225803326614418, 0.0312499696969844,  0.03030300089127989, 0.02941173613446767, 0.02857140079366431,
		        0.0277777507507639,  0.02702700071125037, 0.02631576383267107, 0.02564100064103783, 0.024999975609768,
		        0.02439022009292684, 0.02380950055372122, 0.02325579122622676, 0.02272725050506137, 0.02222220048310243,
		        0.02173910915819728, 0.02127657491135772, 0.02083331292518007, 0.02040814326531593, 0.01999998039216648,
		        0.0196078239064951,  0.01923075036285396, 0.01886790600979246, 0.01851850033670926, 0.0181818003246841,
		        0.01785712531329183, 0.01754384240775197, 0.01724136236120062, 0.01694913587571441, 0.01666665027323211,
		        0.0163934264939265,  0.01612901638505646, 0.01587300024802357, 0.01562498461539219, 0.0153846002331077,
		        0.01515150022614937, 0.01492535842845325, 0.0147058678601947,  0.01449273933748116, 0.01428570020121419,
		        0.01408449315337148, 0.01388887519026551, 0.01369861662347945, 0.01351350018018676, 0.01333332017544509,
		        0.01315788174983553, 0.0129870001665065,  0.01282050016229122, 0.01265821534810744, 0.01249998765432708,
		        0.01234566681722975, 0.01219510990303269, 0.01204818086632832, 0.01190475014006183, 0.01176469425445171,
		        0.011627895482497,   0.01149424150993247, 0.01136362512768686, 0.01123594394507416, 0.01111110012210556,
		        0.01098900011945115, 0.01086955446470845, 0.0107526775337504,  0.01063828734602985, 0.01052630537281217,
		        0.01041665635739342, 0.01030926814643888, 0.01020407153164796, 0.01010100010101505, 0.0099999900990148,
		        0.00990098029509319};

		const Eigen::VectorX<long double> roots = es.eigenvalues();
		const Eigen::VectorX<long double> weights = referenceMoments[0] * es.eigenvectors().row(0).cwiseAbs2();
		std::cout << "Roots: " << std::scientific << std::setprecision(16) << roots.transpose() << std::endl;
		std::cout << "Weights: " << weights.transpose() << std::endl;

		for (Eigen::Index i = 0; i < degree * 2; ++i)
		{
			CHECK(roots.cwisePow(i).dot(weights) == Catch::Approx(referenceMoments[i]).margin(1e-14));
		}
	}

	SECTION("Gaussian 1e-6")
	{
		const auto auxSize = 100;

		QuadratureGrid grid = GenerateFejerQuadratureGrid<long double>(auxSize);
		grid.Nodes() = (grid.Nodes() + Eigen::VectorX<long double>::Constant(auxSize, 1)) / 2;
		for (int i = 0; i < auxSize; ++i)
		{
			grid.Weight(i) *= Exp(-1e-6 * grid.Node(i) * grid.Node(i)) / 2;
		}
		std::cout << "Aux Roots: " << grid.Nodes().transpose() << std::endl;
		std::cout << "Aux Weights: " << grid.Weights().transpose() << std::endl;

		const auto degree = 50;
		const auto jacobiRules = ConstructOrthogonalPolynomialRecurrence(grid, degree);

		std::cout << "Alphas: " << jacobiRules.Alphas.transpose() << std::endl;
		std::cout << "Gammas: " << jacobiRules.Gammas.transpose() << std::endl;

		Eigen::MatrixX<long double> jacobian = Eigen::MatrixX<long double>::Zero(degree, degree);
		jacobian.diagonal() = jacobiRules.Alphas;
		jacobian.diagonal(1) = jacobiRules.Gammas.segment(1, degree - 1);
		jacobian.diagonal(-1) = jacobian.diagonal(1);

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<long double>> es(jacobian);

		// Mathematica: Table[NumberForm[N@Integrate[t^n E^(-SetPrecision[10^-6,1000] t^2),{t,0,1}],16],{n,0,50 2}]
		const std::vector referenceMoments = {
		        0.999999666666767,   0.4999997500000833,  0.3333331333334048,  0.2499998333333958,  0.1999998571429127,
		        0.1666665416667167,  0.1428570317460772,  0.1249999000000417,  0.1111110202020587,  0.0999999166667024,
		        0.0909090139860473,  0.0833332619047932,  0.07692301025643967, 0.07142850892859921, 0.06666660784316357,
		        0.06249994444446944, 0.05882347678020957, 0.05555550555557829, 0.05263153132834254, 0.04999995454547538,
		        0.04761900414080674, 0.04545450378789802, 0.04347822086958374, 0.04166662820514606, 0.0399999629629802,
		        0.03846150274726941, 0.03703700255429454, 0.03571425238096801, 0.03448272636264029, 0.03333330208334804,
		        0.03225803421311302, 0.03124997058824918, 0.03030300173161524, 0.02941173692811773, 0.02857140154441437,
		        0.02777775146200081, 0.02702700138601358, 0.02631576447369611, 0.02564100125079336, 0.02499997619048755,
		        0.02439022064663618, 0.02380950108226195, 0.02325579173127679, 0.02272725098815271, 0.02222220094563668,
		        0.02173910960145927, 0.02127657533652739, 0.02083331333334295, 0.02040814365747242, 0.01999998076924003,
		        0.01960782426933946, 0.01923075071225964, 0.01886790634649248, 0.01851850066138428, 0.01818180063796701,
		        0.01785712561577188, 0.01754384269997846, 0.01724136264368623, 0.0169491361489382,  0.01666665053764222,
		        0.01639342674994264, 0.01612901663307209, 0.01587300048840795, 0.0156249848484922,  0.0153846004592495,
		        0.01515150044563994, 0.01492535864158178, 0.01470586806723384, 0.01449273953868821, 0.01428570039683215,
		        0.01408449334363005, 0.01388887537538195, 0.01369861680365946, 0.01351350035562519, 0.01333332034632667,
		        0.01315788191633553, 0.01298700032879131, 0.01282050032051892, 0.01265821550242828, 0.012499987804884,
		        0.01234566696415879, 0.01219511004646342, 0.0120481810063842,  0.01190475027686061, 0.01176469438810569,
		        0.01162789561311338, 0.01149424163761366, 0.01136362525253069, 0.01123594406717416, 0.01111110024155122,
		        0.01098900023632808, 0.01086955457909864, 0.01075267764573238, 0.01063828745567886, 0.01052630548020038,
		        0.01041665646259003, 0.01030926824951031, 0.01020407163265796, 0.01010100020002486, 0.00999999019608324,
		        0.0099009803902768};

		const Eigen::VectorX<long double> roots = es.eigenvalues();
		const Eigen::VectorX<long double> weights = referenceMoments[0] * es.eigenvectors().row(0).cwiseAbs2();
		std::cout << "Roots: " << std::scientific << std::setprecision(16) << roots.transpose() << std::endl;
		std::cout << "Weights: " << weights.transpose() << std::endl;

		for (Eigen::Index i = 0; i < degree * 2; ++i)
		{
			CHECK(roots.cwisePow(i).dot(weights) == Catch::Approx(referenceMoments[i]).margin(1e-14));
		}
	}
}
