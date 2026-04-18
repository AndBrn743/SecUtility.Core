//
// Created by Andy on 4/16/2026.
//

#include "SecUtility/Diagnostic/TypeName.hpp"
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Math/MatrixFreeLinearOperator.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

TEMPLATE_TEST_CASE("MatrixFreeLinearOperator", "[template]", float, double, (std::complex<double>))
{
	using namespace SecUtility::Math;

	SECTION("Basic square operator")
	{
		const MatrixFreeLinearOperator op([](const Eigen::VectorX<TestType>& vector) { return vector; }, 5);

		STATIC_CHECK(std::is_same_v<typename decltype(op)::Scalar, TestType>);
		CHECK(op.rows() == 5);
		CHECK(op.cols() == 5);

		CHECK(op.ApplyOn(Eigen::Vector<TestType, 5>{4.2, 0., 0.3, 6.9, 0.})[0] == TestType{4.2});
		CHECK((op * Eigen::VectorX<TestType>{{4.2, 0., 0.3, 6.9, 0.}})[0] == TestType{4.2});
		CHECK((op * Eigen::VectorX<typename Eigen::NumTraits<TestType>::Real>{{4.2, 0., 0.3, 6.9, 0.}})[0]
		      == TestType{4.2});

		CHECK(op.ExtractColumn(0) == Eigen::VectorX<TestType>{{1, 0, 0, 0, 0}});
		CHECK(op.ExtractColumn(2) == Eigen::VectorX<TestType>{{0, 0, 1, 0, 0}});
		CHECK(op.ToDense() == Eigen::MatrixX<TestType>::Identity(5, 5));

		{
			const auto op2 = TestType{0.25} * op;
			CHECK(op2.ToDense() == TestType{0.25} * Eigen::MatrixX<TestType>::Identity(5, 5));
		}

		{
			const auto op2 = op * TestType{0.25};
			CHECK(op2.ToDense() == TestType{0.25} * Eigen::MatrixX<TestType>::Identity(5, 5));
		}

		{
			const auto op2 = op / TestType{0.25};
			CHECK(op2.ToDense() == 4 * Eigen::MatrixX<TestType>::Identity(5, 5));
		}

		{
			const auto op2 = op + op;
			CHECK(op2.ToDense() == 2 * Eigen::MatrixX<TestType>::Identity(5, 5));
		}

		{
			const auto op2 = -op;
			CHECK(op2.ToDense() == -Eigen::MatrixX<TestType>::Identity(5, 5));
		}

		{
			const auto op2 = op * op;
			CHECK(op2.ToDense() == Eigen::MatrixX<TestType>::Identity(5, 5));
		}

		{
			const auto op2 = op / 2 + TestType{0.25} * op;
			CHECK(op2.ToDense() == TestType{0.75} * Eigen::MatrixX<TestType>::Identity(5, 5));
		}

		{
			const auto op2 = op - TestType{0.25} * op;
			CHECK(op2.ToDense() == TestType{0.75} * Eigen::MatrixX<TestType>::Identity(5, 5));
		}
	}

	SECTION("Rectangular operators - 3x5")
	{
		// A projection from R^5 to R^3
		const MatrixFreeLinearOperator proj(
		        [](const Eigen::VectorX<TestType>& v) { return Eigen::VectorX<TestType>{{v[0], v[1], v[2]}}; }, 3, 5);

		CHECK(proj.rows() == 3);
		CHECK(proj.cols() == 5);

		CHECK(proj.ToDense() == Eigen::MatrixX<TestType>::Identity(3, 5));

		const Eigen::VectorX<TestType> input{{1, 2, 3, 4, 5}};
		const auto result = proj.ApplyOn(input);
		CHECK(result.size() == 3);
		CHECK(result[0] == TestType{1});
		CHECK(result[1] == TestType{2});
		CHECK(result[2] == TestType{3});
	}

	SECTION("Rectangular operators - 5x3")
	{
		// An embedding from R^3 to R^5
		const MatrixFreeLinearOperator embed([](const Eigen::VectorX<TestType>& v)
		                                     { return Eigen::VectorX<TestType>{{v[0], v[1], v[2], 0, 0}}; },
		                                     5,
		                                     3);

		CHECK(embed.rows() == 5);
		CHECK(embed.cols() == 3);

		CHECK(embed.ToDense() == Eigen::MatrixX<TestType>::Identity(5, 3));

		const Eigen::VectorX<TestType> input{{1, 2, 3}};
		const auto result = embed.ApplyOn(input);
		CHECK(result.size() == 5);
		CHECK(result[0] == TestType{1});
		CHECK(result[1] == TestType{2});
		CHECK(result[2] == TestType{3});
		CHECK(result[3] == TestType{0});
		CHECK(result[4] == TestType{0});
	}

	SECTION("Rectangular operator ToDense and ExtractColumn")
	{
		// 4x2 operator: [[1, 2], [3, 4], [5, 6], [7, 8]]
		const MatrixFreeLinearOperator opRect(
		        [](const Eigen::VectorX<TestType>& v)
		        {
			        return Eigen::VectorX<TestType>{
			                {v[0] + 2 * v[1], 3 * v[0] + 4 * v[1], 5 * v[0] + 6 * v[1], 7 * v[0] + 8 * v[1]}};
		        },
		        4,
		        2);

		auto dense = opRect.ToDense();
		CHECK(dense.rows() == 4);
		CHECK(dense.cols() == 2);

		// Check first column (apply to e1 = [1, 0])
		auto col0 = opRect.ExtractColumn(0);
		CHECK(col0.size() == 4);
		CHECK(col0[0] == TestType{1});
		CHECK(col0[1] == TestType{3});
		CHECK(col0[2] == TestType{5});
		CHECK(col0[3] == TestType{7});

		// Check second column (apply to e2 = [0, 1])
		auto col1 = opRect.ExtractColumn(1);
		CHECK(col1.size() == 4);
		CHECK(col1[0] == TestType{2});
		CHECK(col1[1] == TestType{4});
		CHECK(col1[2] == TestType{6});
		CHECK(col1[3] == TestType{8});
	}

	SECTION("Operator composition with rectangular operators")
	{
		// op1: 3x5 (projection)
		const MatrixFreeLinearOperator op1([](const Eigen::VectorX<TestType>& v)
		                                   { return Eigen::VectorX<TestType>{{v[0] + v[1], v[2] + v[3], v[4]}}; },
		                                   3,
		                                   5);

		// op2: 5x4 (embedding + transformation)
		const MatrixFreeLinearOperator op2([](const Eigen::VectorX<TestType>& v)
		                                   { return Eigen::VectorX<TestType>{{v[0], v[1], v[0] + v[1], v[2], v[3]}}; },
		                                   5,
		                                   4);

		// Composition: op1 * op2 should be 3x4
		const auto composed = op1 * op2;
		CHECK(composed.rows() == 3);
		CHECK(composed.cols() == 4);

		const Eigen::VectorX<TestType> input{{1, 2, 3, 4}};
		const auto result = composed.ApplyOn(input);
		CHECK(result.size() == 3);

		// Manual computation:
		// op2(input) = [1, 2, 3, 3, 4]
		// op1([1, 2, 3, 3, 4]) = [3, 6, 4]
		CHECK(result[0] == TestType{3});
		CHECK(result[1] == TestType{6});
		CHECK(result[2] == TestType{4});
	}

	SECTION("Rectangular operator arithmetic")
	{
		// Two 3x4 operators
		const MatrixFreeLinearOperator op1([](const Eigen::VectorX<TestType>& v)
		                                   { return Eigen::VectorX<TestType>::Constant(3, 1) * v.sum(); },
		                                   3,
		                                   4);

		const MatrixFreeLinearOperator op2([](const Eigen::VectorX<TestType>& v)
		                                   { return Eigen::VectorX<TestType>::Constant(3, 2) * v.sum(); },
		                                   3,
		                                   4);

		// Addition
		const auto sum = op1 + op2;
		CHECK(sum.rows() == 3);
		CHECK(sum.cols() == 4);

		const Eigen::VectorX<TestType> input{{1, 1, 1, 1}};
		const auto sumResult = sum.ApplyOn(input);
		CHECK(sumResult[0] == TestType{12});  // (1+2) * 4

		// Subtraction
		const auto diff = op2 - op1;
		auto diffResult = diff.ApplyOn(input);
		CHECK(diffResult[0] == TestType{4});  // (2-1) * 4
	}

	SECTION("Const correctness")
	{
		const MatrixFreeLinearOperator constOp([](const Eigen::VectorX<TestType>& v) { return v; }, 5);

		// All operations should work on const references
		CHECK(constOp.rows() == 5);
		CHECK(constOp.cols() == 5);

		const Eigen::VectorX<TestType> input{{1, 2, 3, 4, 5}};
		const auto result = constOp.ApplyOn(input);
		CHECK(result[0] == TestType{1});

		auto result2 = constOp * input;
		CHECK(result2[0] == TestType{1});

		auto negated = -constOp;
		CHECK(negated.rows() == 5);
	}

	SECTION("Move semantics")
	{
		struct Func
		{
			auto operator()(const Eigen::VectorX<TestType>& v) const
			{
				return 2 * v;
			}
		};

		auto makeOperator = [] { return MatrixFreeLinearOperator(Func{}, 5); };

		// Test move construction
		{
			auto op1 = makeOperator();
			MatrixFreeLinearOperator op2(std::move(op1));

			CHECK(op2.rows() == 5);
			CHECK(op2.cols() == 5);

			const Eigen::VectorX<TestType> input{{1, 2, 3, 4, 5}};
			const auto result = op2.ApplyOn(input);
			CHECK(result[0] == TestType{2});
		}

		// Test move assignment
		{
			auto op1 = makeOperator();
			auto op2 = makeOperator();
			op2 = std::move(op1);

			CHECK(op2.rows() == 5);
			CHECK(op2.cols() == 5);

			const Eigen::VectorX<TestType> input{{1, 2, 3, 4, 5}};
			const auto result = op2.ApplyOn(input);
			CHECK(result[0] == TestType{2});
		}
	}

	SECTION("Copy-assignment operator")
	{
		struct Func
		{
			auto operator()(const Eigen::VectorX<TestType>& v) const
			{
				return 2 * v;
			}
		};

		auto makeOperator = [] { return MatrixFreeLinearOperator(Func{}, 5); };

		auto op1 = makeOperator();
		auto op2 = makeOperator();
		op2 = op1;

		CHECK(op2.rows() == 5);
		CHECK(op2.cols() == 5);

		const Eigen::VectorX<TestType> input{{1, 2, 3, 4, 5}};
		const auto result = op2.ApplyOn(input);
		CHECK(result[0] == TestType{2});
	}

	SECTION("Stateful functor")
	{
		int applyCount = 0;
		int copyCount = 0;

		auto makeCountingFunctor = [&applyCount, &copyCount]
		{
			struct CountingFunctor
			{
				int* applyCount;
				int* copyCount;

				Eigen::VectorX<TestType> operator()(const Eigen::VectorX<TestType>& v) const
				{
					++(*applyCount);
					return v;
				}

				CountingFunctor(int* a, int* c) : applyCount(a), copyCount(c)
				{
					/* NO CODE */
				}

				CountingFunctor(const CountingFunctor& other) : applyCount(other.applyCount), copyCount(other.copyCount)
				{
					++(*copyCount);
				}

				CountingFunctor(CountingFunctor&&) noexcept = default;
			};

			return CountingFunctor{&applyCount, &copyCount};
		};

		MatrixFreeLinearOperator op(makeCountingFunctor(), 5);

		Eigen::VectorX<TestType> input{{1, 2, 3, 4, 5}};

		applyCount = 0;
		copyCount = 0;

		op.ApplyOn(input);
		CHECK(applyCount == 1);
		CHECK(copyCount == 0);

		// Test with operator composition
		const auto op2 = op + op;
		CHECK(applyCount == 1);
		CHECK(copyCount == 2);
		applyCount = 0;
		copyCount = 0;

		op2.ApplyOn(input);
		CHECK(applyCount == 2);
		CHECK(copyCount == 0);
	}

	SECTION("Complex operator chain")
	{
		// Test a complex expression: (A + B) * C - 0.5 * D
		const MatrixFreeLinearOperator opA(
		        [](const Eigen::VectorX<TestType>& v) { return Eigen::VectorX<TestType>{{v[0], v[1], v[2]}}; }, 3, 3);

		const MatrixFreeLinearOperator opB(
		        [](const Eigen::VectorX<TestType>& v)
		        { return Eigen::VectorX<TestType>{{v[0] + v[1], v[1] + v[2], v[2] + v[0]}}; },
		        3,
		        3);

		const MatrixFreeLinearOperator opC([](const Eigen::VectorX<TestType>& v) { return 2 * v; }, 3, 3);

		const MatrixFreeLinearOperator opD([](const Eigen::VectorX<TestType>& v) { return 3 * v; }, 3, 3);

		const auto complexOp = (opA + opB) * opC - TestType{0.5} * opD;

		CHECK(complexOp.rows() == 3);
		CHECK(complexOp.cols() == 3);

		const Eigen::VectorX<TestType> input{{1, 2, 3}};
		const auto result = complexOp.ApplyOn(input);

		// Manual computation:
		// C * input = [2, 4, 6]
		// (A + B) applied to [2, 4, 6]:
		//   A: [2, 4, 6]
		//   B: [6, 10, 8]
		//   sum: [8, 14, 14]
		// 0.5 * D applied to [1, 2, 3]:
		//   D: [3, 6, 9]
		//   0.5 * D: [1.5, 3, 4.5]
		// Final: [8-1.5, 14-3, 14-4.5] = [6.5, 11, 9.5]
		CHECK(result[0] == TestType{6.5});
		CHECK(result[1] == TestType{11});
		CHECK(result[2] == TestType{9.5});
	}

	SECTION("Scalar multiplication with different scalar types")
	{
		const MatrixFreeLinearOperator op([](const Eigen::VectorX<TestType>& v) { return v; }, 3);

		const auto scaled = TestType{2.5} * op;
		Eigen::VectorX<TestType> input{{1, 2, 3}};

		auto result = scaled.ApplyOn(input);
		CHECK(result[0] == TestType{2.5});
		CHECK(result[1] == TestType{5});
		CHECK(result[2] == TestType{7.5});
	}

	SECTION("Unary plus identity")
	{
		const MatrixFreeLinearOperator op([](const Eigen::VectorX<TestType>& v) { return v; }, 3);

		const auto plusOp = +op;
		Eigen::VectorX<TestType> input{{1, 2, 3}};

		auto result = plusOp.ApplyOn(input);
		CHECK(result[0] == TestType{1});
		CHECK(result[1] == TestType{2});
		CHECK(result[2] == TestType{3});
	}

	SECTION("Mocked complicated linear operator")
	{
		const Eigen::MatrixX<TestType> matrixA = Eigen::MatrixX<TestType>::Random(7, 5);
		const Eigen::MatrixX<TestType> matrixB = Eigen::MatrixX<TestType>::Random(5, 6);
		const Eigen::MatrixX<TestType> matrixC = Eigen::MatrixX<TestType>::Random(7, 6);

		MatrixFreeLinearOperator opA([&matrixA](const Eigen::VectorX<TestType>& v) { return (matrixA * v).eval(); },
		                             matrixA.rows(),
		                             matrixA.cols());
		MatrixFreeLinearOperator opB([&matrixB](const Eigen::VectorX<TestType>& v) { return (matrixB * v).eval(); },
		                             matrixB.rows(),
		                             matrixB.cols());
		MatrixFreeLinearOperator opC([&matrixC](const Eigen::VectorX<TestType>& v) { return (matrixC * v).eval(); },
		                             matrixC.rows(),
		                             matrixC.cols());

		const Eigen::MatrixX<TestType> matrixD = TestType{0.69} * matrixA * matrixB - matrixC / TestType{0.42};
		const auto opD = TestType{0.69} * opA * opB - opC / TestType{0.42};

		CHECK(matrixD.rows() == opD.rows());
		CHECK(matrixD.cols() == opD.cols());

		CHECK((opD.ToDense() - matrixD).norm()
		      < 30 * std::numeric_limits<typename Eigen::NumTraits<TestType>::Real>::epsilon());

		const Eigen::VectorX<TestType> v = Eigen::VectorX<TestType>::Random(opD.cols());
		CHECK((opD.ApplyOn(v) - matrixD * v).norm()
		      < 30 * std::numeric_limits<typename Eigen::NumTraits<TestType>::Real>::epsilon());
	}
}
