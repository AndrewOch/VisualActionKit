import CoreML

@available(macOS 10.16, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
class ActionsInput : MLFeatureProvider {
    var Placeholder: MLMultiArray

    var featureNames: Set<String> {
        get {
            return ["Placeholder"]
        }
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        if (featureName == "Placeholder") {
            return MLFeatureValue(multiArray: Placeholder)
        }
        return nil
    }
    
    init(Placeholder: MLMultiArray) {
        self.Placeholder = Placeholder
    }
}

@available(macOS 10.16, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
class ActionsOutput : MLFeatureProvider {

    private let provider : MLFeatureProvider

    lazy var Softmax: [String : Double] = {
        [unowned self] in return self.provider.featureValue(for: "Softmax")!.dictionaryValue as! [String : Double]
    }()
    lazy var classLabel: String = {
        [unowned self] in return self.provider.featureValue(for: "classLabel")!.stringValue
    }()

    var featureNames: Set<String> {
        return self.provider.featureNames
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        return self.provider.featureValue(for: featureName)
    }

    init(Softmax: [String : Double], classLabel: String) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["Softmax" : MLFeatureValue(dictionary: Softmax as [AnyHashable : NSNumber]), "classLabel" : MLFeatureValue(string: classLabel)])
    }

    init(features: MLFeatureProvider) {
        self.provider = features
    }
}

@available(macOS 10.16, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
class Actions {
    let model: MLModel
    class var urlOfModelInThisBundle : URL {
        let bundle = Bundle(for: self)
        return bundle.url(forResource: "ActionsML", withExtension:"mlmodelc")!
    }
    init(model: MLModel) {
        self.model = model
    }
    @available(*, deprecated, message: "Use init(configuration:) instead and handle errors appropriately.")
    convenience init() {
        try! self.init(contentsOf: type(of:self).urlOfModelInThisBundle)
    }
    convenience init(configuration: MLModelConfiguration) throws {
        try self.init(contentsOf: type(of:self).urlOfModelInThisBundle, configuration: configuration)
    }
    convenience init(contentsOf modelURL: URL) throws {
        try self.init(model: MLModel(contentsOf: modelURL))
    }
    convenience init(contentsOf modelURL: URL, configuration: MLModelConfiguration) throws {
        try self.init(model: MLModel(contentsOf: modelURL, configuration: configuration))
    }
    class func load(configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Result<Actions, Error>) -> Void) {
        return self.load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration, completionHandler: handler)
    }
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Result<Actions, Error>) -> Void) {
        MLModel.__loadContents(of: modelURL, configuration: configuration) { (model, error) in
            if let error = error {
                handler(.failure(error))
            } else if let model = model {
                handler(.success(Actions(model: model)))
            } else {
                fatalError("SPI failure: -[MLModel loadContentsOfURL:configuration::completionHandler:] vends nil for both model and error.")
            }
        }
    }
    func prediction(input: ActionsInput) throws -> ActionsOutput {
        return try self.prediction(input: input, options: MLPredictionOptions())
    }
    func prediction(input: ActionsInput, options: MLPredictionOptions) throws -> ActionsOutput {
        let outFeatures = try model.prediction(from: input, options:options)
        return ActionsOutput(features: outFeatures)
    }
    func prediction(Placeholder: MLMultiArray) throws -> ActionsOutput {
        let input_ = ActionsInput(Placeholder: Placeholder)
        return try self.prediction(input: input_)
    }
    func predictions(inputs: [ActionsInput], options: MLPredictionOptions = MLPredictionOptions()) throws -> [ActionsOutput] {
        let batchIn = MLArrayBatchProvider(array: inputs)
        let batchOut = try model.predictions(from: batchIn, options: options)
        var results : [ActionsOutput] = []
        results.reserveCapacity(inputs.count)
        for i in 0..<batchOut.count {
            let outProvider = batchOut.features(at: i)
            let result =  ActionsOutput(features: outProvider)
            results.append(result)
        }
        return results
    }
}
