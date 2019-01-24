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

            Catchrate = Variable.Random<double, Gaussian>(CatchratePrior);
        }

        public void SetModelData(AnimalModelData Priors, Gaussian CatchrateDist)
        {
            base.SetModelData(Priors);
            this.CatchratePrior.ObservedValue = CatchrateDist;
        }

        public Variable<double> GetCatchableMouse()
        {
            return Catchrate * Population;
        }
    }
}