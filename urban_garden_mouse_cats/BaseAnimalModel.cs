using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace catandmouse
{
    public class BaseAnimalModel
    {
        // parameter to set
        protected Variable<double> initialPopulation;

        // best guesses
        protected Variable<double> BornYoungPerLitter;
        protected Variable<double> Birthrate;
        protected Variable<double> Deathrate;

        // prior
        protected Variable<Gaussian> BornYoungPerLitterPrior;
        protected Variable<Gaussian> BirthratePrior;
        protected Variable<Gaussian> DeathratePrior;

        public virtual void CreateModel()
        {
            BornYoungPerLitterPrior = Variable.New<Gaussian>();
            BirthratePrior = Variable.New<Gaussian>();
            DeathratePrior = Variable.New<Gaussian>();

            BornYoungPerLitter = Variable.Random<double, Gaussian>(BornYoungPerLitterPrior);
            Birthrate = Variable.Random<double, Gaussian>(BirthratePrior);
            Deathrate = Variable.Random<double, Gaussian>(DeathratePrior);
        }

        public virtual void SetModelData(AnimalModelData priors)
        {
            BornYoungPerLitterPrior.ObservedValue = priors.BornYoungPerLitterDist;
            BirthratePrior.ObservedValue = priors.BirthrateDist;
            DeathratePrior.ObservedValue = priors.DeathrateDist;
        }

        public struct AnimalModelData
        {
            public Gaussian BornYoungPerLitterDist;
            public Gaussian BirthrateDist;
            public Gaussian DeathrateDist;

            public AnimalModelData(
                Gaussian BornYoungPerLitterDist,
                Gaussian BirthrateDist,
                Gaussian DeathrateDist
            ){
                this.BornYoungPerLitterDist = BornYoungPerLitterDist;
                this.BirthrateDist = BirthrateDist;
                this.DeathrateDist = DeathrateDist;
            }
        }
    }
}