feature {
  name: "aid"
  type: INT
  int_domain {
    name: "aid"
    min: 0
    max: 1855610 
    is_categorical: true
  }
  annotation {
    tag: "item_id"
    tag: "list"
    tag: "categorical"
    tag: "item"
  }
  value_count {
    min: 2
    max: 20
  }
}

feature {
  name: "target"
  type: INT
  int_domain {
    name: "target"
    min: 0
    max: 2 
    is_categorical: true
  }
  annotation {
    tag: "list"
    tag: "categorical"
    tag: "item"
  }
  value_count {
    min: 2
    max: 20
  }
}

feature {
  name: "ts"
  type: INT
  int_domain {
    name: "ts"
    min: 0
    max: 2 
    is_categorical: false
  }
  annotation {
    tag: "list"
    tag: "item"
  }
  value_count {
    min: 2
    max: 20
  }
}
