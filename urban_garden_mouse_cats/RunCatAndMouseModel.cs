using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace catandmouse
{
    class Program
    {
        static void Main(string[] args)
        {
            // Cat();
            // Mouse();
            CatsAndMice();
            
        }

        public static void Mouse()
        {
            BaseAnimalModel.AnimalModelData MousePriors = new BaseAnimalModel.AnimalModelData(
                new Gaussian(5.5, 2.22),
                new Gaussian(0.0195, 0.000004),
                new Gaussian(0.001141552511, 0.0000000225),
                60
            );

            BaseAnimalModel Mouse = new BaseAnimalModel();
            Mouse.CreateModel();
            Mouse.SetModelData(MousePriors);

            Gaussian BornYoung = Mouse.InferBornYoung();
            Console.WriteLine(
                "BornYoung Mean: {0:f10}, Standard Deviation: {1:f10}",
                BornYoung.GetMean(),
                Math.Sqrt(BornYoung.GetVariance())
            );

            Gaussian NaturalDeath = Mouse.InferNaturalDeath();
            Console.WriteLine(
                "NaturalDeath Mean: {0:f10}, Standard Deviation: {1:f10}",
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
            CatModel.CatModelData CatSpecificPriors = new CatModel.CatModelData(
                new Gaussian(13.5, 30.25),
                new Gaussian(3.5, 2.25)
            );

            CatModel Cat = new CatModel();
            Cat.CreateModel();
            Cat.SetModelData(CatPriors, CatSpecificPriors);

            Gaussian CatchableMouse = Cat.InferCatchableMouse();
            Console.WriteLine(
                "Catchable Mouse Mean: {0:f10}, Standard Deviation: {1:f10}",
                CatchableMouse.GetMean(),
                Math.Sqrt(CatchableMouse.GetVariance())
            );
            Gaussian FoodNeeds = Cat.InferFoodNeeds();
            Console.WriteLine(
                "FoodNeeds Mouse Mean: {0:f10}, Standard Deviation: {1:f10}",
                FoodNeeds.GetMean(),
                Math.Sqrt(FoodNeeds.GetVariance())
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
                1
            );
            CatModel.CatModelData CatSpecificPriors = new CatModel.CatModelData(
                new Gaussian(13.5, 30.25),
                new Gaussian(3.5, 2.25)
            );
            CatAndMouse Cats = new CatAndMouse();

            Cats.CreateModel();
            Cats.SetModelData(
                MousePriors,
                CatPriors,
                CatSpecificPriors
            );

            Gaussian CatchedMouse = Cats.InferCatchedMouse();
            Gaussian StarvingCats = Cats.InferStarvingCats();
            Gaussian DyingCats = Cats.InferDyingCats();
            Gaussian DyingMouse = Cats.InferDyingMouse();
            Gaussian CatPopulationChange = Cats.InferCatPopulationChange();
            Gaussian MousePopulationChange = Cats.InferMousePopulationChange();
            CatAndMouse.CatAndMousePopulation NewPopulation = Cats.InferPopulation();
            
            
            Console.WriteLine(
                "CatchedMouse Mean: {0:f2}, Standard Deviation: {1:f2}",
                CatchedMouse.GetMean(),
                Math.Sqrt(CatchedMouse.GetVariance())
            );
            Console.WriteLine(
                "StarvingCats Mean: {0:f2}, Standard Deviation: {1:f2}",
                StarvingCats.GetMean(),
                Math.Sqrt(StarvingCats.GetVariance())
            );
            Console.WriteLine(
                "DyingCats Mean: {0:f2}, Standard Deviation: {1:f2}",
                DyingCats.GetMean(),
                Math.Sqrt(DyingCats.GetVariance())
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
    }
}