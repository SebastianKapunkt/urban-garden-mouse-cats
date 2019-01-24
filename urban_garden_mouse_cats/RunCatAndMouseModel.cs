using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using System.Collections;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Algorithms;

namespace catandmouse
{
    class Program
    {
        static void Main(string[] args)
        {
            // Cat();
            // Mouse();
            // CatsAndMice();
            ContiniousView();
        }

        public static void Mouse()
        {
            BaseAnimalModel.AnimalModelData MousePriors = new BaseAnimalModel.AnimalModelData(
                new Gaussian(5.5, 2.22),
                new Gaussian(0.0195, 0.000004),
                new Gaussian(0.001141552511, 0.0000000225)
            );

            InferenceEngine engine = new InferenceEngine();

            BaseAnimalModel Mouse = new BaseAnimalModel();
            Mouse.CreateModel();
            Mouse.SetModelData(MousePriors);
            Mouse.SetNewPopulation(100);

            Gaussian BornYoung = engine.Infer<Gaussian>(Mouse.GetBornYoung());
            Console.WriteLine(
                "Mouse BornYoung Mean: {0:f10}, Standard Deviation: {1:f10}",
                BornYoung.GetMean(),
                Math.Sqrt(BornYoung.GetVariance())
            );

            Gaussian NaturalDeath = engine.Infer<Gaussian>(Mouse.GetNaturalDeath());
            Console.WriteLine(
                "Mouse NaturalDeath Mean: {0:f10}, Standard Deviation: {1:f10}",
                NaturalDeath.GetMean(),
                Math.Sqrt(NaturalDeath.GetVariance())
            );
        }

        public static void Cat()
        {
            BaseAnimalModel.AnimalModelData CatPriors = new BaseAnimalModel.AnimalModelData(
                new Gaussian(5.5, 4),
                new Gaussian(0.004102103451, 0.00000144),
                new Gaussian(0.0007305936073, 0.0000000225)
            );
            Gaussian Catchrate = new Gaussian(13.5, 30.25);

            InferenceEngine engine = new InferenceEngine();

            CatModel Cat = new CatModel();
            Cat.CreateModel();
            Cat.SetModelData(CatPriors, Catchrate);
            Cat.SetNewPopulation(10);

            Gaussian CatchableMouse = engine.Infer<Gaussian>(Cat.GetCatchableMouse());
            Console.WriteLine(
                "Catchable Mouse Mean: {0:f10}, Standard Deviation: {1:f10}",
                CatchableMouse.GetMean(),
                Math.Sqrt(CatchableMouse.GetVariance())
            );
        }

        public static void CatsAndMice()
        {
            BaseAnimalModel.AnimalModelData MousePriors = new BaseAnimalModel.AnimalModelData(
                new Gaussian(5.5, 2.22),
                new Gaussian(0.0195, 0.000004),
                new Gaussian(0.001141552511, 0.0000000225)
            );
            BaseAnimalModel.AnimalModelData CatPriors = new BaseAnimalModel.AnimalModelData(
                new Gaussian(5.5, 4),
                new Gaussian(0.004102103451, 0.00000144),
                new Gaussian(0.0007305936073, 0.0000000225)
            );
            Gaussian Catchrate = new Gaussian(13.5, 30.25);
            CatAndMouse Model = new CatAndMouse();

            Model.CreateModel();
            Model.SetModelData(
                MousePriors,
                CatPriors,
                Catchrate
            );
            Model.Cat.SetNewPopulation(10);
            Model.Mouse.SetNewPopulation(100);

            Gaussian MouseBornYoung = Model.engine.Infer<Gaussian>(Model.Mouse.GetBornYoung());
            Gaussian MouseNaturalDeath = Model.engine.Infer<Gaussian>(Model.Mouse.GetNaturalDeath());
            Gaussian CatBornYoung = Model.engine.Infer<Gaussian>(Model.Cat.GetBornYoung());
            Gaussian CatNaturalDeath = Model.engine.Infer<Gaussian>(Model.Cat.GetNaturalDeath());
            Gaussian CatchedMouse = Model.engine.Infer<Gaussian>(Model.GetCatchedMouse());
            Gaussian DyingMouse = Model.engine.Infer<Gaussian>(Model.GetDyingMouse());
            Gaussian CatPopulationChange = Model.engine.Infer<Gaussian>(Model.GetCatPopulationChange());
            Gaussian MousePopulationChange = Model.engine.Infer<Gaussian>(Model.GetMousePopulationChange());

            Console.WriteLine(
                "initial Mouse Population: {0:f2}, initial Cat Population: {1:f2}",
                Model.engine.Infer<Gaussian>(Model.Mouse.Population).GetMean(),
                Model.engine.Infer<Gaussian>(Model.Cat.Population).GetMean()
            );
            Console.WriteLine(
                "Mouse NaturalDeath Mean: {0:f10}, Standard Deviation: {1:f10}",
                MouseNaturalDeath.GetMean(),
                Math.Sqrt(MouseNaturalDeath.GetVariance())
            );
            Console.WriteLine(
                "Mouse BornYoung Mean: {0:f10}, Standard Deviation: {1:f10}",
                MouseBornYoung.GetMean(),
                Math.Sqrt(MouseBornYoung.GetVariance())
            );
            Console.WriteLine(
                "Cat NaturalDeath Mean: {0:f10}, Standard Deviation: {1:f10}",
                CatNaturalDeath.GetMean(),
                Math.Sqrt(CatNaturalDeath.GetVariance())
            );
            Console.WriteLine(
                "Cat BornYoung Mean: {0:f10}, Standard Deviation: {1:f10}",
                CatBornYoung.GetMean(),
                Math.Sqrt(CatBornYoung.GetVariance())
            );
            Console.WriteLine(
                "CatchedMouse Mean: {0:f2}, Standard Deviation: {1:f2}",
                CatchedMouse.GetMean(),
                Math.Sqrt(CatchedMouse.GetVariance())
            );
            Console.WriteLine(
                "DyingMouse Mean: {0:f2}, Standard Deviation: {1:f2}",
                DyingMouse.GetMean(),
                Math.Sqrt(DyingMouse.GetVariance())
            );
            Console.WriteLine(
                "CatPopulationChange Mean: {0:f2}, Standard Deviation: {1:f2}",
                CatPopulationChange.GetMean(),
                Math.Sqrt(CatPopulationChange.GetVariance())
            );
            Console.WriteLine(
                "MousePopulationChange Mean: {0:f2}, Standard Deviation: {1:f2}",
                MousePopulationChange.GetMean(),
                Math.Sqrt(MousePopulationChange.GetVariance())
            );
        }

        public static void ContiniousView()
        {
            double CurrentCatPopulation = 5;
            double CurrentMousePopulation = 1300;
            int iterations = 1000;

            List<double> CatPopulation = new List<double>();
            List<double> CatPopulationStandardDeviation = new List<double>();

            List<double> MousePopulation = new List<double>();
            List<double> MousePopulationStandardDeviation = new List<double>();

            BaseAnimalModel.AnimalModelData MousePriors = new BaseAnimalModel.AnimalModelData(
                    new Gaussian(5.5, 2.22),
                    new Gaussian(0.0195, 0.000004),
                    new Gaussian(0.001141552511, 0.0000000225)
                );
            BaseAnimalModel.AnimalModelData CatPriors = new BaseAnimalModel.AnimalModelData(
                new Gaussian(5.5, 4),
                new Gaussian(0.004102103451, 0.00000144),
                new Gaussian(0.0007305936073, 0.0000000225)
            );
            Gaussian Catchrate = new Gaussian(13.5, 30.25);

            InferenceEngine engine = new InferenceEngine();
            engine.Algorithm = new ExpectationPropagation();

            CatAndMouse Model = new CatAndMouse();

            Model.CreateModel();
            Model.SetModelData(
                MousePriors,
                CatPriors,
                Catchrate
            );
            Model.Cat.SetNewPopulation(CurrentCatPopulation);
            Model.Mouse.SetNewPopulation(CurrentMousePopulation);
            Variable<double> ruleCat = Model.GetCatPopulationChange();
            Variable<double> ruleMouse = Model.GetMousePopulationChange();

            for (int i = 0; i < iterations; i++)
            {
                Gaussian CatPopulationChange = engine.Infer<Gaussian>(ruleCat);
                Gaussian MousePopulationChange = engine.Infer<Gaussian>(ruleMouse);

                CurrentCatPopulation = CurrentCatPopulation + CatPopulationChange.GetMean();
                CurrentMousePopulation = CurrentMousePopulation + MousePopulationChange.GetMean();

                CatPopulation.Add(CurrentCatPopulation);
                CatPopulationStandardDeviation.Add(Math.Sqrt(CatPopulationChange.GetVariance()));

                MousePopulation.Add(CurrentMousePopulation);
                MousePopulationStandardDeviation.Add(Math.Sqrt(MousePopulationChange.GetVariance()));

                Model.Cat.SetNewPopulation(CurrentCatPopulation);
                Model.Mouse.SetNewPopulation(CurrentMousePopulation);
            }

            for (int i = 0; i < iterations; i++)
            {
                Console.WriteLine(
                    "{0:d2} CatPopulation Mean: {1:f2}, Standard Deviation: {2:f2}",
                    i,
                    CatPopulation[i],
                    CatPopulationStandardDeviation[i]
                );
                Console.WriteLine(
                    "{0:d2} MousePopulation Mean: {1:f2}, Standard Deviation: {2:f2}",
                    i,
                    MousePopulation[i],
                    MousePopulationStandardDeviation[i]
                );
            }
        }
    }
}