using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace catandmouse
{
    public class BaseAnimalModel
    {
        public InferenceEngine engine;

        public Variable<double> Population;
        public Variable<Gaussian> BornYoungPerLitterPrior;
        public Variable<Gaussian> BirthratePrior;
        public Variable<Gaussian> DeathratePrior;

        public Variable<double> BornYoungPerLitter;
        public Variable<double> Birthrate;
        public Variable<double> Deathrate;

        public BaseAnimalModel(InferenceEngine engine){
            this.engine = engine;
        }

        public virtual void CreateModel()
        {
            BornYoungPerLitterPrior = Variable.New<Gaussian>();
            BirthratePrior = Variable.New<Gaussian>();
            DeathratePrior = Variable.New<Gaussian>();
            Population = Variable.New<double>();

            BornYoungPerLitter = Variable.Random<double, Gaussian>(BornYoungPerLitterPrior);
            Birthrate = Variable.Random<double, Gaussian>(BirthratePrior);
            Deathrate = Variable.Random<double, Gaussian>(DeathratePrior);
        }

        public virtual void SetModelData(AnimalModelData priors)
        {
            BornYoungPerLitterPrior.ObservedValue = priors.BornYoungPerLitterDist;
            BirthratePrior.ObservedValue = priors.BirthrateDist;
            DeathratePrior.ObservedValue = priors.DeathrateDist;
            Population.ObservedValue = priors.Population;
        }

        public void SetNewPopulation(double NewPopulation){
            Population.ObservedValue = NewPopulation;
        }

        public Gaussian InferNaturalDeath(){
            return engine.Infer<Gaussian>(Deathrate * Population);
        }

        public double InferFemine(){
            return engine.Infer<Gaussian>(Population * 0.5).GetMean();
        }

        public Gaussian InferBornYoung(){
            Gaussian NumberOfBirthDist = engine.Infer<Gaussian>(Birthrate * InferFemine());
            Variable<double> NumberOfBirth = Variable.Random<double, Gaussian>(NumberOfBirthDist);
            return engine.Infer<Gaussian>(BornYoungPerLitter * NumberOfBirth);
        }

        public struct AnimalModelData
        {
            public Gaussian BornYoungPerLitterDist;
            public Gaussian BirthrateDist;
            public Gaussian DeathrateDist;
            public double Population;

            public AnimalModelData(
                Gaussian BornYoungPerLitterDist,
                Gaussian BirthrateDist,
                Gaussian DeathrateDist,
                double Population
            )
            {
                this.BornYoungPerLitterDist = BornYoungPerLitterDist;
                this.BirthrateDist = BirthrateDist;
                this.DeathrateDist = DeathrateDist;
                this.Population = Population;
            }
        }
    }
}