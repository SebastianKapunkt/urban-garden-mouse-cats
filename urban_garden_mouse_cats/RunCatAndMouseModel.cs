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
            //CatsAndMice();
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
            CatModel.CatModelData CatSpecificPriors = new CatModel.CatModelData(
                new Gaussian(13.5, 30.25),
                new Gaussian(3.5, 2.25)
            );

            CatModel Cat = new CatModel(new InferenceEngine());
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
                "Cat FoodNeeds Mouse Mean: {0:f10}, Standard Deviation: {1:f10}",
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
                10
            );
            CatModel.CatModelData CatSpecificPriors = new CatModel.CatModelData(
                new Gaussian(13.5, 30.25),
                new Gaussian(3.5, 2.25)
            );
            CatAndMouse Model = new CatAndMouse();

            Model.CreateModel();
            Model.SetModelData(
                MousePriors,
                CatPriors,
                CatSpecificPriors
            );

            Gaussian BornYoung = Model.Mouse.InferBornYoung();
            Gaussian NaturalDeath = Model.Mouse.InferNaturalDeath();
            Gaussian CatchedMouse = Model.InferCatchedMouse();
            Gaussian StarvingCats = Model.InferStarvingCats();
            Gaussian DyingCats = Model.InferDyingCats();
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

        public static void ContiniousView()
        {
            Variable<double> initalMousePopulation = 3000;
            Variable<double> initialCatPopulation = 10;

            BaseAnimalModel.AnimalModelData MousePriors = new BaseAnimalModel.AnimalModelData(
                new Gaussian(5.5, 2.22),
                new Gaussian(0.0195, 0.000004),
                new Gaussian(0.001141552511, 0.0000000225),
                initalMousePopulation
            );
            BaseAnimalModel.AnimalModelData CatPriors = new BaseAnimalModel.AnimalModelData(
                new Gaussian(5.5, 4),
                new Gaussian(0.004102103451, 0.00000144),
                new Gaussian(0.0007305936073, 0.0000000225),
                initialCatPopulation
            );
            CatModel.CatModelData CatSpecificPriors = new CatModel.CatModelData(
                new Gaussian(13.5, 30.25),
                new Gaussian(3.5, 2.25)
            );
            CatAndMouse Model = new CatAndMouse();

            Model.CreateModel();
            Model.SetModelData(
                MousePriors,
                CatPriors,
                CatSpecificPriors
            );

            for (int i = 0; i < 10; i++)
            {
                CatAndMouse.CatAndMousePopulation NewPopulation = Model.InferPopulation();
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
                Model.Cat.Population = Variable.Random<double, Gaussian>(NewPopulation.CatPopulation);
                Model.Mouse.Population = Variable.Random<double, Gaussian>(NewPopulation.MousePopulation);
            }
        }
    }
}