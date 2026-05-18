//
// Created by Andy on 5/16/2026.
//

#include <SecUtility/Math/OrthogonalPolynomial.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iomanip>
#include <iostream>


TEST_CASE("Orthogonal polynomial roots and weights")
{
	using namespace SecUtility::Math;

	SECTION("Shifted Legendre")
	{
		const auto auxSize = 100;
		QuadratureGrid grid = FejerQuadratureGrid01<long double>(auxSize);

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

		QuadratureGrid grid = FejerQuadratureGrid01<long double>(auxSize);
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

	SECTION("Exp 1 1CG")
	{
		const auto auxSize = 100;

		QuadratureGrid grid = FirstKindOfChebyshevGaussQuadratureGrid<long double>(auxSize);
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

		// Mathematica: Table[NumberForm[NIntegrate[SetPrecision[t^n Exp[-t]/Sqrt[1-t^2],1000],{t,-1,1}],16],{n,0,50 2}]
		const std::vector referenceMoments = {
		        3.977463260506422,   -1.77549968921215,   2.20196357129427,    -1.349035807130149,  1.705855528328249,
		        -1.131504407699918,  1.444476718349377,   -0.993921739244926,  1.276050989834117,   -0.896850849018481,
		        1.155767262625695,   -0.823637131270877,  1.064267308828031,   -0.7658693142201553, 0.991611799214239,
		        -0.7187791399861069, 0.932095098513078,   -0.6794345519819833, 0.882173954597097,   -0.6459200439565906,
		        0.839515055097592,   -0.6169240793496336, 0.802510267887124,   -0.5915143428823614, 0.7700101272847713,
		        -0.5690074494517216, 0.74116812016843,    -0.5488895071918471, 0.7153451117333271,  -0.5307656232063191,
		        0.6920482401218049,  -0.514326659818682,  0.6708904819322601,  -0.4993266817623233, 0.6515630979734271,
		        -0.4855672544304208, 0.6338163728277148,  -0.4728862617026683, 0.6174458493241523,  -0.4611497820103478,
		        0.6022822956220106,  -0.4502460807718871, 0.5881842643885744,  -0.4400810969542261, 0.5750324877761361,
		        -0.430575003512963,  0.5627255956787574,  -0.4216595522162766, 0.551176803110081,   -0.413275999834013,
		        0.5403113176247405,  -0.405373470990981,  0.5300642887853719,  -0.3979076530134416, 0.5203791706039985,
		        -0.3908397460222489, 0.5112064021061732,  -0.3841356112964901, 0.502502335452199,   -0.3777650751208364,
		        0.4942283585193007,  -0.3717013556425797, 0.486350171572137,   -0.3659205878474409, 0.4788371870233159,
		        -0.360401427399893,  0.4716620282665063,  -0.3551247183260287, 0.4648001088155457,  -0.3500732127248728,
		        0.4582292769692939,  -0.3452313331469783, 0.4519295142749511,  -0.3405849701703777, 0.4458826784198425,
		        -0.3361213091734924, 0.4400722830158799,  -0.3318286814548421, 0.4344833081784829,  -0.3276964357559496,
		        0.4291020369361639,  -0.3237148269630144, 0.423915913407992,   -0.3198749193369762, 0.4189134194061648,
		        -0.3161685020825007, 0.4140839666996278,  -0.3125880154385323, 0.4094178026423988,  -0.3091264857750901,
		        0.4049059272502362,  -0.3057774684273497, 0.4005400201195531,  -0.3025349971999864, 0.3963123758369939,
		        -0.2993935396410304, 0.392215846737849,   -0.2963479573219765, 0.3882437920450842,  -0.2933934704750884,
		        0.3843900325650442};
		const Eigen::VectorX<long double> roots = es.eigenvalues();
		const Eigen::VectorX<long double> weights = referenceMoments[0] * es.eigenvectors().row(0).cwiseAbs2();
		std::cout << "Roots: " << std::scientific << std::setprecision(16) << roots.transpose() << std::endl;
		std::cout << "Weights: " << weights.transpose() << std::endl;

		for (Eigen::Index i = 0; i < degree * 2; ++i)
		{
			CHECK(roots.cwisePow(i).dot(weights) == Catch::Approx(referenceMoments[i]).margin(1e-14));
		}
	}

	SECTION("Exp 1 2CG")
	{
		const auto auxSize = 100;

		QuadratureGrid grid = SecondKindOfChebyshevGaussQuadratureGrid<long double>(auxSize);
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

		// Mathematica: Table[NumberForm[NIntegrate[SetPrecision[t^n Exp[-t] Sqrt[1-t^2],1000],{t,-1,1}],16],{n,0,50 2}]
		const std::vector referenceMoments = {
		        1.775499689212177,    -0.4264638820820491,   0.4961080429660219,   -0.2175313994302471,
		        0.2613788099788862,   -0.1375826684549266,   0.1684257285153879,   -0.0970708902264452,
		        0.1202837272084222,   -0.07321371774734546,  0.0914999537978936,   -0.0577678170505831,
		        0.07265550961376677,  -0.04709017423467791,  0.05951670070059509,  -0.03934458800412378,
		        0.04992114391598091,  -0.03351450802539235,  0.04265889949950545,  -0.02899596460695714,
		        0.03700478721046768,  -0.0254097364672722,   0.0325001406023524,   -0.0225068934306401,
		        0.02884200711634134,  -0.02011794225987418,  0.02582300843510291,  -0.01812388398552789,
		        0.02329687161152228,  -0.01643896338763734,  0.02115775818954448,  -0.01499997805635848,
		        0.01932738395883332,  -0.0137594273319026,   0.01774672513274623,  -0.01268099272775237,
		        0.01637052350356228,  -0.01173647969232066,  0.0151635537021416,   -0.01090370123846075,
		        0.01409803123343624,  -0.01016498381766111,  0.01315177661243828,  -0.00950609344126282,
		        0.01230689209737876,  -0.00891545129668653,  0.01154879256867641,  -0.00838355238226361,
		        0.01086548548534048,  -0.007902528843031867, 0.01024702883936877,  -0.007465817977539423,
		        0.00968511818137324,  -0.007067906991192691, 0.00917276849782528,  -0.006704134725758887,
		        0.00870406665397406,  -0.006370536175653784, 0.00827397693289864,  -0.00606371947825663,
		        0.00787818694716351,  -0.005780767795138728, 0.007512984548821323, -0.005519160447547886,
		        0.007175158756809495, -0.005276709073864378, 0.006861919450960665, -0.005051505601155908,
		        0.006570831846251791, -0.004841879577894515, 0.006299762694342601, -0.004646362976600683,
		        0.006046835855108648, -0.004463660996885267, 0.00581039540396271,  -0.004292627718650327,
		        0.005588974837396933, -0.00413224569889244,  0.005381271242319013, -0.003981608792935279,
		        0.00518612352817194,  -0.003839907626038157, 0.005002494001827257, -0.003706417254475522,
		        0.004829452706536862, -0.003580486643968411, 0.004666164057229087, -0.003461529663442132,
		        0.00451187539216258,  -0.003349017347740519, 0.004365907130683121, -0.003242471227363126,
		        0.00422764428255927,  -0.003141457558956022, 0.004096529099144938, -0.003045582319054087,
		        0.003972054692764646, -0.002954486846888052, 0.003853759480040059, -0.002867844041052455,
		        0.003741222328771218};
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

		QuadratureGrid grid = FejerQuadratureGrid<long double>(auxSize);
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

		QuadratureGrid grid = FejerQuadratureGrid<long double>(auxSize);
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

		QuadratureGrid grid = FejerQuadratureGrid<long double>(auxSize);
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

		QuadratureGrid grid = FejerQuadratureGrid<long double>(auxSize);
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

		QuadratureGrid grid = FejerQuadratureGrid<long double>(auxSize);
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

		QuadratureGrid grid = FejerQuadratureGrid<long double>(auxSize);
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

	SECTION("Exp(x^3 / 3)")
	{
		const auto auxSize = 100;

		QuadratureGrid grid = FejerQuadratureGrid01<long double>(auxSize);
		for (int i = 0; i < auxSize; ++i)
		{
			grid.Weight(i) *= Exp(-PowInt(grid.Node(i), 3) / 3);
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

		// Mathematica: Table[NumberForm[N@Integrate[SetPrecision[t^n E^(- t^3/3),100],{t,0,1}],16],{n,0,50 2}]
		const std::vector referenceMoments = {
		        0.924023413085615,    0.4397514278304956,   0.2834686894262108,   0.2074921025118261,
		        0.1629715450872019,   0.133874757704843,    0.113437099473515,    0.0983264148622204,
		        0.0867172356552688,   0.0775283857408155,   0.07008000832397365,  0.06392381032362939,
		        0.05875254683436571,  0.05434878098992096,  0.0505544133097636,   0.0472517982729651,
		        0.04435162328510401,  0.04178488907266468,  0.03949746179365232,  0.03744628527297893,
		        0.03559669273417498,  0.03392046350560469,  0.03239439488578948,  0.03099923684388539,
		        0.02971888654951394,  0.0285397717993688,   0.02745037367946,     0.02644085316405942,
		        0.02550275620979943,  0.0246287787716307,   0.02381257801987457,  0.0230486195103941,
		        0.0223320525751318,   0.02165860804232234,  0.02102451375882199,  0.02042642440556018,
		        0.01986136286517022,  0.01932667098498035,  0.01881996802637742,  0.01833911543750903,
		        0.01788218685546397,  0.01744744245493028,  0.01703330692657215,  0.01663835050023343,
		        0.01626127253328251,  0.01590088726881331,  0.01555611143648186,  0.01522595342392366,
		        0.01490950379162292,  0.01460592694085792,  0.01431445377454656,  0.01403437521573395,
		        0.01376503646910668,  0.01350583192808541,  0.01325620064437632,  0.01301562228886475,
		        0.01278361354282257,  0.01255972486690848,  0.01234353760263682,  0.01213466136709699,
		        0.01193273170690244,  0.01173740798178307,  0.01154837145203019,  0.01136532354725979,
		        0.01118798429676101,  0.01101609090411245,  0.01084939645083717,  0.0106876687156763,
		        0.01053068909763244,  0.01037825163230085,  0.01023016209219931,  0.01008623716284882,
		        0.00994630368727019,  0.00981019797236158,  0.00967776515132569,  0.00954885859693476,
		        0.00942333938096784,  0.00930107577563778,  0.00918194279325254,  0.00906582176073447,
		        0.00895259992595747,  0.00884217009316125,  0.00873443028496816,  0.00862928342876615,
		        0.00852663706543312,  0.00842640307856851,  0.00832849744256749,  0.00823283998802613,
		        0.00813935418310255,  0.00804796692958271,  0.007958608372509943, 0.007871211722337666,
		        0.007785713088654875, 0.007702051324615412, 0.007620167881275925, 0.007540006671114131,
		        0.007461513940059536, 0.007384638147423748, 0.007309329853167355, 0.007235541611985806,
		        0.007163227873737974};

		const Eigen::VectorX<long double> roots = es.eigenvalues();
		const Eigen::VectorX<long double> weights = referenceMoments[0] * es.eigenvectors().row(0).cwiseAbs2();
		std::cout << "Roots: " << std::scientific << std::setprecision(16) << roots.transpose() << std::endl;
		std::cout << "Weights: " << weights.transpose() << std::endl;

		for (Eigen::Index i = 0; i < degree * 2; ++i)
		{
			CHECK(roots.cwisePow(i).dot(weights) == Catch::Approx(referenceMoments[i]).margin(1e-14));
		}
	}

	SECTION("Exp(x^3 / 3) in [-1, 1]")
	{
		const auto auxSize = 100;

		QuadratureGrid grid = FejerQuadratureGrid<long double>(auxSize);
		for (int i = 0; i < auxSize; ++i)
		{
			grid.Weight(i) *= Exp(-PowInt(grid.Node(i), 3) / 3);
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

		// Mathematica: Table[NumberForm[Re@N@Integrate[SetPrecision[t^n E^(- t^3/3),100],{t,-1,1}],16],{n,0,50 2}]
		const std::vector referenceMoments = {
		        2.01595235569085,    -0.1344597102233058,   0.6790811145123002,  -0.096191379969029,
		        0.4101616940656887,  -0.07490039212297795,  0.2943155946361843,  -0.0613352653314352,
		        0.2296787617744326,  -0.05193457320658879,  0.1883989918608186,  -0.04503487968998527,
		        0.1597353824464124,  -0.03975482519087434,  0.138662558232477,   -0.03558376385651826,
		        0.1225135618400596,  -0.03220536217272416,  0.109740892808008,   -0.02941318437886593,
		        0.0993845954032653,  -0.02706677230772567,  0.0908174269349818,  -0.02506723219130706,
		        0.0836121237423355,  -0.02334291615529855,  0.07746754192093096, -0.02184064210149099,
		        0.07216529447453802, -0.02052010379474292,  0.06754313567055268, -0.01935019589827662,
		        0.06347800067001276, -0.01830652987274574,  0.05987484576744828, -0.01736971354945789,
		        0.05665909883894507, -0.01652413379918881,  0.05377142673181624, -0.01575707861891153,
		        0.0511640301431254,  -0.01505809311904548,  0.04879796975583917, -0.0144184997917373,
		        0.04664120351239003, -0.01383103615879408,  0.04466712367585917, -0.01328957760232757,
		        0.04285345120777274, -0.01278892289449772,  0.0411813896005769,  -0.0123246264790142,
		        0.03963496978741424, -0.01189286603045664,  0.03820053760356177, -0.01149033692692377,
		        0.0368663488676417,  -0.0111141674639814,   0.03562224660456923, -0.01076185020430168,
		        0.03445940160137893, -0.01043118599029458,  0.03337010225419972, -0.01012023797576431,
		        0.03234758311403623, -0.00982729364529632,  0.0313858840633841,  -0.00955083324752412,
		        0.03047973392274348, -0.00928950341314358,  0.02962445368065994, -0.00904209499057875,
		        0.02881587559224945, -0.0088075243330229,   0.0280502751906305,  -0.00858481742566903,
		        0.02732431386860573, -0.0083730963625913,   0.02663499016145423, -0.00817156777723789,
		        0.02597959823017905, -0.007979512904994884, 0.0253556923332691,  -0.007796279015375972,
		        0.02476105630271983, -0.007621271998543145, 0.02419367722071866, -0.00745394992869306,
		        0.02365172263758979, -0.007293817457355839, 0.02313352078731095, -0.007140420914387287,
		        0.02263754335027476, -0.006993344014582608, 0.0221623903886699,  -0.006852204084325803,
		        0.02170677714153508, -0.006716648736238633, 0.02126952241702318, -0.00658635293097622,
		        0.02084953836091429};

		const Eigen::VectorX<long double> roots = es.eigenvalues();
		const Eigen::VectorX<long double> weights = referenceMoments[0] * es.eigenvectors().row(0).cwiseAbs2();
		std::cout << "Roots: " << std::scientific << std::setprecision(16) << roots.transpose() << std::endl;
		std::cout << "Weights: " << weights.transpose() << std::endl;

		for (Eigen::Index i = 0; i < degree * 2; ++i)
		{
			CHECK(roots.cwisePow(i).dot(weights) == Catch::Approx(referenceMoments[i]).margin(1e-14));
		}
	}

	SECTION("Exp(x^2 / 4) in [-1, 1]")
	{
		const auto auxSize = 100;

		QuadratureGrid grid = FejerQuadratureGrid<long double>(auxSize);
		for (int i = 0; i < auxSize; ++i)
		{
			grid.Weight(i) *= Exp(-PowInt(grid.Node(i), 2) / 4);
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

		// Mathematica: Table[NumberForm[Re@N@Integrate[SetPrecision[t^n E^(- t^2/4),100],{t,-1,1}],16],{n,0,50 2}]
		const std::vector referenceMoments = {
		        1.84512402565117,    0., 0.5750449190167202,  0., 0.3350663818147012,  0., 0.2354606858613928,  0.,
		        0.18124646977388,    0., 0.1472333236442209,  0., 0.123929987887241,   0., 0.1069765527826478,  0.,
		        0.0940934511938151,  0., 0.0839742083040938,  0., 0.07581678326994329, 0., 0.06910176505199884, 0.,
		        0.06347806010632677, 0., 0.05869987303071918, 0., 0.05459001137321609, 0., 0.05101752736091385, 0.,
		        0.04788356409103942, 0., 0.04511209772298235, 0., 0.04264370832314459, 0., 0.04043128362707998, 0.,
		        0.0384369906266188,  0., 0.0366300990971226,  0., 0.03498539006692438, 0., 0.03348197373757478, 0.,
		        0.03210239904641019, 0., 0.03083197426257849, 0., 0.02965824249738605, 0., 0.02857057243730234, 0.,
		        0.02755983581763835, 0., 0.02661815092515227, 0., 0.02573867688234811, 0., 0.0249154473608502,  0.,
		        0.02414323518150573, 0., 0.02341744131012493, 0., 0.02273400327112062, 0., 0.02208931912902634, 0.,
		        0.02148018403612102, 0., 0.0209037369880494,  0., 0.02035741592179001, 0., 0.01983891967004217, 0.,
		        0.01934617558104405, 0., 0.01887731184351598, 0., 0.01843063373803304, 0., 0.01800460317999797, 0.,
		        0.01759782103402843, 0., 0.01720901177144033, 0., 0.0168370101165206,  0., 0.01648074938721164, 0.,
		        0.01613925128459291, 0., 0.0158116169254051,  0., 0.01549701894459126};

		const Eigen::VectorX<long double> roots = es.eigenvalues();
		const Eigen::VectorX<long double> weights = referenceMoments[0] * es.eigenvectors().row(0).cwiseAbs2();
		std::cout << "Roots: " << std::scientific << std::setprecision(16) << roots.transpose() << std::endl;
		std::cout << "Weights: " << weights.transpose() << std::endl;

		for (Eigen::Index i = 0; i < degree * 2; ++i)
		{
			CHECK(roots.cwisePow(i).dot(weights) == Catch::Approx(referenceMoments[i]).margin(1e-14));
		}
	}

	SECTION("Exp(-x) / (1 + Exp(-x))^2 in [0, 1]")
	{
		const auto auxSize = 100;

		QuadratureGrid grid = FejerQuadratureGrid01<long double>(auxSize);
		for (int i = 0; i < auxSize; ++i)
		{
			const auto exp = Exp(-grid.Node(i));
			grid.Weight(i) *= exp / PowInt(1 + exp, 2);
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

		// Mathematica: Table[NumberForm[Re@NIntegrate[SetPrecision[t^n (E^-t
		// /(SetPrecision[1,1000]+E^-t)^2),1000],{t,0,1}],16],{n,0,50 2}]
		const std::vector referenceMoments = {
		        0.2310585786300051,   0.1109440716717275,   0.07217327763488139,  0.05325232608595112,
		        0.04210803243025896,  0.03478524384106956,  0.02961484565775342,  0.02577346851700999,
		        0.02280903847476376,  0.02045308871456653,  0.01853631416167172,  0.01694674211638776,
		        0.01560740046274459,  0.01446365586048499,  0.01347567066799296,  0.01261371921704276,
		        0.01185517528120039,  0.01118251476587821,  0.01058195605912802,  0.01004251274026857,
		        0.00955531988378068,  0.00911314607647871,  0.00871003409017943,  0.0083410323312631,
		        0.00800199141150862,  0.007689408144749871, 0.007400304560718397, 0.007132133102147069,
		        0.006882701628208472, 0.006650113561755202, 0.006432719730796564, 0.006229079323946458,
		        0.006037928010056337, 0.005858151734619668, 0.005688765048152532, 0.005528893078105645,
		        0.005377756449415745, 0.005234658606193737, 0.005098975100186637, 0.004970144499158533,
		        0.004847660636503722, 0.004731065976867386, 0.004619945914746773, 0.004513923856555338,
		        0.004412656963395934, 0.004315832453282759, 0.004223164378903229, 0.004134390811088381,
		        0.004049271369636243, 0.003967585052529754, 0.003889128322319758, 0.003813713414826587,
		        0.00374116684060642,  0.003671328054033954, 0.003604048268533136, 0.003539189399573107,
		        0.00347662311964186,  0.003416230011600541, 0.00335789880867572,  0.003301525710921605,
		        0.003247013769325325, 0.003194272329873686, 0.003143216530880756, 0.003093766847717725,
		        0.003045848679811521, 0.002999391975404249, 0.002954330890106597, 0.002910603475747563,
		        0.002868151396430444, 0.002826919669060018, 0.00278685642591559,  0.002747912697115425,
		        0.002710042211055364, 0.002673201211112716, 0.002637348287089694, 0.002602444220032051,
		        0.00256845183920097,  0.002535335890102262, 0.002503062912588292, 0.002471601128147184,
		        0.002440920335581414, 0.002410991814356443, 0.002381788234969391, 0.002353283575750112,
		        0.002325453045562523, 0.002298273011923768, 0.002271720934103343, 0.002245775300804314,
		        0.002220415572064635, 0.002195622125048933, 0.002171376203430189, 0.002147659870087007,
		        0.002124455962865871, 0.002101748053179162, 0.002079520407229145, 0.002057757949665643,
		        0.002036446229501053, 0.00201557138812083,  0.001995120129240661, 0.001975079690673526,
		        0.001955437817780714};

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
