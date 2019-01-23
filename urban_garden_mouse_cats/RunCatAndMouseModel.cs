using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using System.Collections;
using System.Collections.Generic;

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
                new Gaussian(0.001141552511, 0.0000000225),
                60
            );

            BaseAnimalModel Mouse = new BaseAnimalModel(new InferenceEngine());
            Mouse.CreateModel();
            Mouse.SetModelData(MousePriors);

            Gaussian BornYoung = Mouse.InferBornYoung();
            Console.WriteLine(
                "Mouse BornYoung Mean: {0:f10}, Standard Deviation: {1:f10}",
                BornYoung.GetMean(),
                Math.Sqrt(BornYoung.GetVariance())
            );

            Gaussian NaturalDeath = Mouse.InferNaturalDeath();
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
                new Gaussian(0.0007305936073, 0.0000000225),
                2
            );
            Gaussian Catchrate = new Gaussian(13.5, 30.25);

            CatModel Cat = new CatModel(new InferenceEngine());
            Cat.CreateModel();
            Cat.SetModelData(CatPriors, Catchrate);

            Gaussian CatchableMouse = Cat.InferCatchableMouse();
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
                new Gaussian(0.001141552511, 0.0000000225),
                3000
            );
            BaseAnimalModel.AnimalModelData CatPriors = new BaseAnimalModel.AnimalModelData(
                new Gaussian(5.5, 4),
                new Gaussian(0.004102103451, 0.00000144),
                new Gaussian(0.0007305936073, 0.0000000225),
                10
            );
            Gaussian Catchrate = new Gaussian(13.5, 30.25);
            CatAndMouse Model = new CatAndMouse();

            Model.CreateModel();
            Model.SetModelData(
                MousePriors,
                CatPriors,
                Catchrate
            );

            Gaussian BornYoung = Model.Mouse.InferBornYoung();
            Gaussian NaturalDeath = Model.Mouse.InferNaturalDeath();
            Gaussian CatchedMouse = Model.InferCatchedMouse();
            Gaussian DyingMouse = Model.InferDyingMouse();
            Gaussian CatPopulationChange = Model.InferCatPopulationChange();
            Gaussian MousePopulationChange = Model.InferMousePopulationChange();
            CatAndMouse.CatAndMousePopulation NewPopulation = Model.InferPopulation();

            Console.WriteLine(
                "initial Mouse Population: {0:f2}, initial Cat Population: {1:f2}",
                Model.engine.Infer<Gaussian>(Model.Mouse.Population).GetMean(),
                Model.engine.Infer<Gaussian>(Model.Cat.Population).GetMean()
            );
            Console.WriteLine(
                "Mouse NaturalDeath Mean: {0:f10}, Standard Deviation: {1:f10}",
                NaturalDeath.GetMean(),
                Math.Sqrt(NaturalDeath.GetVariance())
            );
            Console.WriteLine(
                "Mouse BornYoung Mean: {0:f10}, Standard Deviation: {1:f10}",
                BornYoung.GetMean(),
                Math.Sqrt(BornYoung.GetVariance())
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
            Console.WriteLine(
                "New CatPopulation Mean: {0:f2}, Standard Deviation: {1:f2}",
                NewPopulation.CatPopulation.GetMean(),
                Math.Sqrt(NewPopulation.CatPopulation.GetVariance())
            );
            Console.WriteLine(
                "New MousePopulation Mean: {0:f2}, Standard Deviation: {1:f2}",
                NewPopulation.MousePopulation.GetMean(),
                Math.Sqrt(NewPopulation.MousePopulation.GetVariance())
            );
        }

        public static void ContiniousView()
        {
            double LatestCatPopulation = 5;
            double LatestMousePopulation = 500;

            List<double> CatPopulation = new List<double>();
            List<double> CatPopulationStandardDeviation = new List<double>();

            List<double> MousePopulation = new List<double>();
            List<double> MousePopulationStandardDeviation = new List<double>();

            for (int i = 0; i < 20; i++)
            {
                Variable<double> CurrentCatPopulation = LatestCatPopulation;
                Variable<double> CurrentMousePopulation = LatestMousePopulation;
                BaseAnimalModel.AnimalModelData MousePriors = new BaseAnimalModel.AnimalModelData(
                    new Gaussian(5.5, 2.22),
                    new Gaussian(0.0195, 0.000004),
                    new Gaussian(0.001141552511, 0.0000000225),
                    CurrentMousePopulation
                );
                BaseAnimalModel.AnimalModelData CatPriors = new BaseAnimalModel.AnimalModelData(
                    new Gaussian(5.5, 4),
                    new Gaussian(0.004102103451, 0.00000144),
                    new Gaussian(0.0007305936073, 0.0000000225),
                    CurrentCatPopulation
                );
                Gaussian Catchrate = new Gaussian(13.5, 30.25);
                CatAndMouse Model = new CatAndMouse();

                Model.CreateModel();
                Model.SetModelData(
                    MousePriors,
                    CatPriors,
                    Catchrate
                );

                CatAndMouse.CatAndMousePopulation NewPopulation = Model.InferPopulation();

                LatestCatPopulation = NewPopulation.CatPopulation.GetMean();
                LatestMousePopulation = NewPopulation.MousePopulation.GetMean();

                CatPopulation.Add(LatestCatPopulation);
                CatPopulationStandardDeviation.Add(Math.Sqrt(NewPopulation.CatPopulation.GetVariance()));
                MousePopulation.Add(LatestMousePopulation);
                MousePopulationStandardDeviation.Add(Math.Sqrt(NewPopulation.MousePopulation.GetVariance()));
            }

            for (int i = 0; i < 20; i++)
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