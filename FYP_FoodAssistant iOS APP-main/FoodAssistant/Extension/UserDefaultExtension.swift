//
//  UserDefaultExtension.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 24/12/2022.
//

import Foundation
import Combine

public protocol AnyOptional {
    var isNil: Bool { get }
}

extension Optional: AnyOptional {
    public var isNil: Bool {
        switch self {
        case .none:
            return true
        case .some:
            return false
        }
    }
}

@propertyWrapper
struct UserDefaultExtension<Value> {
    let key: String
    let defaultValue: Value
    let userDefaults: UserDefaults
    private let publisher = PassthroughSubject<Value, Never>()

    var wrappedValue: Value {
        get { userDefaults.object(forKey: key) as? Value ?? defaultValue }
        set { 
            if let value = newValue as? AnyOptional, value.isNil {
                userDefaults.removeObject(forKey: key)
            } else {
                userDefaults.set(newValue, forKey: key)
            }
            publisher.send(newValue)
        }
    }

    var projectedValue: AnyPublisher<Value, Never> {
        publisher.eraseToAnyPublisher()
    }
}

extension UserDefaultExtension where Value: ExpressibleByNilLiteral {
    init(key: String, userDefaults: UserDefaults = .standard) {
        self.init(key: key, defaultValue: nil, userDefaults: userDefaults)
    }
}
