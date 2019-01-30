using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace catandmouse
{

    public class CatModel : BaseAnimalModel
    {
        private Variable<double> Catchrate;
        private Variable<Gaussian> CatchratePrior;

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
            return Catchrate * base.GetPopulation();
        }
    }
}