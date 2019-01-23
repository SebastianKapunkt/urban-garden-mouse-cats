using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;

namespace catandmouse
{

    public class CatAndMouse
    {
        public BaseAnimalModel Mouse;
        public CatModel Cat;
        public InferenceEngine engine;

        public virtual void CreateModel()
        {
            if (engine == null)
            {
                engine = new InferenceEngine();
            }

            Mouse = new BaseAnimalModel(engine);
            Cat = new CatModel(engine);

            Mouse.CreateModel();
            Cat.CreateModel();
        }

        public void SetModelData(
            BaseAnimalModel.AnimalModelData MousePriors,
            BaseAnimalModel.AnimalModelData CatPriors,
            Gaussian CatchrateDist
        )
        {
            Mouse.SetModelData(MousePriors);
            Cat.SetModelData(CatPriors, CatchrateDist);
        }

        public Gaussian InferCatchedMouse()
        {
            Variable<double> CatchedMouse = Variable.New<double>();

            Variable<double> CatchableMouse = Variable.Random<double, Gaussian>(Cat.InferCatchableMouse());
            Variable<double> MousePopulation = Mouse.Population;

            Variable<bool> condition = CatchableMouse > MousePopulation;
            using (Variable.If(condition))
            {
                CatchedMouse.SetTo(Mouse.Population);
            }
            using (Variable.IfNot(condition))
            {
                CatchedMouse.SetTo(CatchableMouse);
            }

            return engine.Infer<Gaussian>(CatchedMouse);
        }

        public Gaussian InferDyingMouse()
        {
            Variable<double> Catched = Variable.Random<double, Gaussian>(InferCatchedMouse());
            Variable<double> NaturalDeath = Variable.Random<double, Gaussian>(Mouse.InferNaturalDeath());

            return engine.Infer<Gaussian>(Catched + NaturalDeath);
        }

        public Gaussian InferCatPopulationChange()
        {
            Variable<double> Deceased = Variable.Random<double, Gaussian>(Cat.InferNaturalDeath());
            Variable<double> BornYoung = Variable.Random<double, Gaussian>(Cat.InferBornYoung());

            return engine.Infer<Gaussian>(BornYoung - Deceased);
        }

        public Gaussian InferMousePopulationChange()
        {
            Variable<double> Deceased = Variable.Random<double, Gaussian>(InferDyingMouse());
            Variable<double> BornYoung = Variable.Random<double, Gaussian>(Mouse.InferBornYoung());

            return engine.Infer<Gaussian>(BornYoung - Deceased);
        }

        public CatAndMousePopulation InferPopulation()
        {
            Variable<double> CatPopulationChange = Variable.Random<double, Gaussian>(InferCatPopulationChange());
            Variable<double> MousePopulationChange = Variable.Random<double, Gaussian>(InferMousePopulationChange());
            Variable<double> CatPopulation = Cat.Population;
            Variable<double> MousePopulation = Mouse.Population;

            Gaussian NewCatPopulation = engine.Infer<Gaussian>(CatPopulation + CatPopulationChange);
            Gaussian NewMousePopulation = engine.Infer<Gaussian>(MousePopulation + MousePopulationChange);

            CatAndMousePopulation CatAndMouse = new CatAndMousePopulation(
                NewCatPopulation,
                NewMousePopulation
            );

            return CatAndMouse;
        }

        public struct CatAndMousePopulation
        {
            public Gaussian CatPopulation;
            public Gaussian MousePopulation;

            public CatAndMousePopulation(Gaussian Cats, Gaussian Mice)
            {
                this.CatPopulation = Cats;
                this.MousePopulation = Mice;
            }
        }
    }
}