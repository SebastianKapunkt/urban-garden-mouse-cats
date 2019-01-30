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

            Gaussian BornYoung = engine.Infer<Gaussian>(Mouse.GetBornYoung(0.5));
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
            InferenceEngine engine = new InferenceEngine();

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
            Model.Cat.SetNewPopulation(5);
            Model.Mouse.SetNewPopulation(1000);

            Gaussian MouseBornYoung = engine.Infer<Gaussian>(Model.Mouse.GetBornYoung(0.5));
            Gaussian MouseNaturalDeath = engine.Infer<Gaussian>(Model.Mouse.GetNaturalDeath());
            Gaussian CatBornYoung = engine.Infer<Gaussian>(Model.Cat.GetBornYoung(0.5));
            Gaussian CatNaturalDeath = engine.Infer<Gaussian>(Model.Cat.GetNaturalDeath());
            Gaussian CatchedMouse = engine.Infer<Gaussian>(Model.GetCatchedMouse());
            Gaussian DyingMouse = engine.Infer<Gaussian>(Model.GetDyingMouse());
            Gaussian CatPopulationChange = engine.Infer<Gaussian>(Model.GetCatPopulationChange());
            Gaussian MousePopulationChange = engine.Infer<Gaussian>(Model.GetMousePopulationChange());

            Console.WriteLine(
                "initial Mouse Population: {0:f2}, initial Cat Population: {1:f2}",
                engine.Infer<Gaussian>(Model.Mouse.GetPopulation()).GetMean(),
                engine.Infer<Gaussian>(Model.Cat.GetPopulation()).GetMean()
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
            double InitialCatPopilation = 0;
            double InitialMousePopulation = 2000;
            int Iterations = 365;

            InferenceEngine engine = new InferenceEngine();

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

            Console.WriteLine(
                engine.Infer(Model.GetPopulationForIteration(Iterations, InitialCatPopilation, InitialMousePopulation))
            );
        }
    }
}
