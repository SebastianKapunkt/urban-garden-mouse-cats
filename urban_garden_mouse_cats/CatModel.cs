using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace catandmouse
{

    public class CatModel : BaseAnimalModel
    {
        public Variable<double> Catchrate;
        public Variable<double> FoodNeeds;

        public Variable<Gaussian> CatchratePrior;
        public Variable<Gaussian> FoodNeedsPrior;

        public override void CreateModel()
        {
            base.CreateModel();

            CatchratePrior = Variable.New<Gaussian>();
            FoodNeedsPrior = Variable.New<Gaussian>();

            Catchrate = Variable.Random<double, Gaussian>(CatchratePrior);
            FoodNeeds = Variable.Random<double, Gaussian>(FoodNeedsPrior);
        }

        public void SetModelData(AnimalModelData Priors, CatModelData CatPriors)
        {
            base.SetModelData(Priors);
            this.CatchratePrior.ObservedValue = CatPriors.CatchrateDist;
            this.FoodNeedsPrior.ObservedValue = CatPriors.FoodNeeds;
        }

        public Gaussian InferCatchableMouse()
        {
            return engine.Infer<Gaussian>(Catchrate * Population);
        }

        public Gaussian InferFoodNeeds()
        {
            return engine.Infer<Gaussian>(FoodNeeds * Population);
        }

        public struct CatModelData
        {
            public Gaussian CatchrateDist;
            public Gaussian FoodNeeds;

            public CatModelData(Gaussian CatchrateDist, Gaussian FoodNeeds)
            {
                this.CatchrateDist = CatchrateDist;
                this.FoodNeeds = FoodNeeds;
            }
        }
    }
}