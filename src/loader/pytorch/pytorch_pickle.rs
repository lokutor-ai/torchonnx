use std::collections::HashMap;
use std::io::{Read, Cursor};
use crate::loader::LoaderError;
use byteorder::{ReadBytesExt, LittleEndian, BigEndian};

#[derive(Debug, Clone)]
pub enum PickleValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
    List(Vec<PickleValue>),
    Tuple(Vec<PickleValue>),
    Dict(HashMap<String, PickleValue>),
    Object {
        class_name: String,
        module_name: String,
        dict: Option<HashMap<String, PickleValue>>,
    },
    PersistentId(Box<PickleValue>),
}

pub struct PytorchUnpickler<'a> {
    reader: Cursor<&'a [u8]>,
    stack: Vec<PickleValue>,
    memo: HashMap<i64, PickleValue>,
}

impl<'a> PytorchUnpickler<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            reader: Cursor::new(data),
            stack: Vec::new(),
            memo: HashMap::new(),
        }
    }

    fn read_line(&mut self) -> Result<String, LoaderError> {
        let mut buf = Vec::new();
        loop {
            let b = self.reader.read_u8()?;
            if b == b'\n' { break; }
            buf.push(b);
        }
        String::from_utf8(buf).map_err(|e| LoaderError::InvalidFormat(e.to_string()))
    }

    pub fn parse(&mut self) -> Result<PickleValue, LoaderError> {
        loop {
            let opcode = self.reader.read_u8()?;
            match opcode {
                0x80 => {
                    let _version = self.reader.read_u8()?;
                }
                b'N' => {
                    self.stack.push(PickleValue::None);
                }
                b'I' => {
                    let line = self.read_line()?;
                    let val = line.parse::<i64>().map_err(|e| LoaderError::InvalidFormat(e.to_string()))?;
                    self.stack.push(PickleValue::Int(val));
                }
                b'J' => {
                    let val = self.reader.read_i32::<LittleEndian>()?;
                    self.stack.push(PickleValue::Int(val as i64));
                }
                b'K' => {
                    let val = self.reader.read_u8()?;
                    self.stack.push(PickleValue::Int(val as i64));
                }
                b'M' => {
                    let val = self.reader.read_u16::<LittleEndian>()?;
                    self.stack.push(PickleValue::Int(val as i64));
                }
                b'F' => {
                    let line = self.read_line()?;
                    let val = line.parse::<f64>().map_err(|e| LoaderError::InvalidFormat(e.to_string()))?;
                    self.stack.push(PickleValue::Float(val));
                }
                b'G' => {
                    let val = self.reader.read_f64::<BigEndian>()?;
                    self.stack.push(PickleValue::Float(val));
                }
                b'S' => {
                    let line = self.read_line()?;
                    let val = if (line.starts_with('\'') && line.ends_with('\'')) || (line.starts_with('"') && line.ends_with('"')) {
                        line[1..line.len()-1].to_string()
                    } else {
                        line
                    };
                    self.stack.push(PickleValue::String(val));
                }
                b'T' => {
                    let len = self.reader.read_u32::<LittleEndian>()?;
                    let mut buf = vec![0u8; len as usize];
                    self.reader.read_exact(&mut buf)?;
                    let val = String::from_utf8(buf).map_err(|e| LoaderError::InvalidFormat(e.to_string()))?;
                    self.stack.push(PickleValue::String(val));
                }
                b'U' => {
                    let len = self.reader.read_u8()?;
                    let mut buf = vec![0u8; len as usize];
                    self.reader.read_exact(&mut buf)?;
                    let val = String::from_utf8(buf).map_err(|e| LoaderError::InvalidFormat(e.to_string()))?;
                    self.stack.push(PickleValue::String(val));
                }
                b'V' => {
                    let line = self.read_line()?;
                    self.stack.push(PickleValue::String(line));
                }
                b'X' => {
                    let len = self.reader.read_u32::<LittleEndian>()?;
                    let mut buf = vec![0u8; len as usize];
                    self.reader.read_exact(&mut buf)?;
                    let val = String::from_utf8(buf).map_err(|e| LoaderError::InvalidFormat(e.to_string()))?;
                    self.stack.push(PickleValue::String(val));
                }
                b'Q' => {
                    let pid = self.stack.pop().ok_or(LoaderError::InvalidFormat("Stack empty for BINPERSID".to_string()))?;
                    self.stack.push(PickleValue::PersistentId(Box::new(pid)));
                }
                b'P' => {
                    let line = self.read_line()?;
                    self.stack.push(PickleValue::PersistentId(Box::new(PickleValue::String(line))));
                }
                b'(' => {
                    self.stack.push(PickleValue::String("__MARK__".to_string()));
                }
                b't' => {
                    let mut items = Vec::new();
                    while let Some(val) = self.stack.pop() {
                        if let PickleValue::String(ref s) = val {
                            if s == "__MARK__" { break; }
                        }
                        items.push(val);
                    }
                    items.reverse();
                    self.stack.push(PickleValue::Tuple(items));
                }
                b']' => {
                    self.stack.push(PickleValue::List(Vec::new()));
                }
                b'a' => {
                    let val = self.stack.pop().ok_or(LoaderError::InvalidFormat("Stack empty for APPEND".to_string()))?;
                    if let Some(PickleValue::List(list)) = self.stack.last_mut() {
                        list.push(val);
                    } else {
                        return Err(LoaderError::InvalidFormat("APPEND expect list on top".to_string()));
                    }
                }
                b'e' => {
                    let mut items = Vec::new();
                    while let Some(val) = self.stack.pop() {
                        if let PickleValue::String(ref s) = val {
                            if s == "__MARK__" { break; }
                        }
                        items.push(val);
                    }
                    items.reverse();
                    if let Some(PickleValue::List(list)) = self.stack.last_mut() {
                        list.extend(items);
                    } else {
                        return Err(LoaderError::InvalidFormat("APPENDS expect list on top".to_string()));
                    }
                }
                b'}' => {
                    self.stack.push(PickleValue::Dict(HashMap::new()));
                }
                b's' => {
                    let val = self.stack.pop().ok_or(LoaderError::InvalidFormat("Stack empty for SETITEM (val)".to_string()))?;
                    let key = self.stack.pop().ok_or(LoaderError::InvalidFormat("Stack empty for SETITEM (key)".to_string()))?;
                    if let Some(PickleValue::Dict(dict)) = self.stack.last_mut() {
                        if let PickleValue::String(k) = key {
                            dict.insert(k, val);
                        }
                    }
                }
                b'u' => {
                    let mut items = Vec::new();
                    while let Some(val) = self.stack.pop() {
                        if let PickleValue::String(ref s) = val {
                            if s == "__MARK__" { break; }
                        }
                        items.push(val);
                    }
                    items.reverse();
                    if let Some(PickleValue::Dict(dict)) = self.stack.last_mut() {
                        for i in (0..items.len()).step_by(2) {
                            if i + 1 < items.len() {
                                if let PickleValue::String(ref k) = items[i] {
                                    dict.insert(k.clone(), items[i+1].clone());
                                }
                            }
                        }
                    }
                }
                b'q' => {
                    let memo_idx = self.reader.read_u8()?;
                    let val = self.stack.last().ok_or(LoaderError::InvalidFormat("Stack empty for BINPUT".to_string()))?.clone();
                    self.memo.insert(memo_idx as i64, val);
                }
                b'r' => {
                    let memo_idx = self.reader.read_u32::<LittleEndian>()?;
                    let val = self.stack.last().ok_or(LoaderError::InvalidFormat("Stack empty for LONG_BINPUT".to_string()))?.clone();
                    self.memo.insert(memo_idx as i64, val);
                }
                b'h' => { // BINGET
                    let memo_idx = self.reader.read_u8()?;
                    let val = self.memo.get(&(memo_idx as i64)).ok_or(LoaderError::InvalidFormat(format!("Memo index {} not found", memo_idx)))?.clone();
                    self.stack.push(val);
                }
                b'j' => { // LONG_BINGET
                    let memo_idx = self.reader.read_u32::<LittleEndian>()?;
                    let val = self.memo.get(&(memo_idx as i64)).ok_or(LoaderError::InvalidFormat(format!("Memo index {} not found", memo_idx)))?.clone();
                    self.stack.push(val);
                }
                b'c' => { // GLOBAL
                    let module_name = self.read_line()?;
                    let class_name = self.read_line()?;
                    self.stack.push(PickleValue::Object {
                        module_name,
                        class_name,
                        dict: None,
                    });
                }
                b'R' => { // REDUCE
                    let _args = self.stack.pop().ok_or(LoaderError::InvalidFormat("Stack empty for REDUCE (args)".to_string()))?;
                    let obj = self.stack.pop().ok_or(LoaderError::InvalidFormat("Stack empty for REDUCE (obj)".to_string()))?;
                    self.stack.push(obj); // Minimal REDUCE: just keep the object
                }
                b'b' => { // BUILD
                    let dict_val = self.stack.pop().ok_or(LoaderError::InvalidFormat("Stack empty for BUILD".to_string()))?;
                    if let PickleValue::Dict(dict) = dict_val {
                        if let Some(PickleValue::Object { dict: obj_dict, .. }) = self.stack.last_mut() {
                            *obj_dict = Some(dict);
                        }
                    }
                }
                0x85 => { // TUPLE1
                    let val = self.stack.pop().ok_or(LoaderError::InvalidFormat("Stack empty for TUPLE1".to_string()))?;
                    self.stack.push(PickleValue::Tuple(vec![val]));
                }
                0x86 => { // TUPLE2
                    let val2 = self.stack.pop().ok_or(LoaderError::InvalidFormat("Stack empty for TUPLE2".to_string()))?;
                    let val1 = self.stack.pop().ok_or(LoaderError::InvalidFormat("Stack empty for TUPLE2".to_string()))?;
                    self.stack.push(PickleValue::Tuple(vec![val1, val2]));
                }
                0x87 => { // TUPLE3
                    let val3 = self.stack.pop().ok_or(LoaderError::InvalidFormat("Stack empty for TUPLE3".to_string()))?;
                    let val2 = self.stack.pop().ok_or(LoaderError::InvalidFormat("Stack empty for TUPLE3".to_string()))?;
                    let val1 = self.stack.pop().ok_or(LoaderError::InvalidFormat("Stack empty for TUPLE3".to_string()))?;
                    self.stack.push(PickleValue::Tuple(vec![val1, val2, val3]));
                }
                0x88 => { // NEWTRUE
                    self.stack.push(PickleValue::Bool(true));
                }
                0x89 => { // NEWFALSE
                    self.stack.push(PickleValue::Bool(false));
                }
                0x2e => {
                    return self.stack.pop().ok_or(LoaderError::InvalidFormat("Stack empty at STOP".to_string()));
                }
                _ => {
                    return Err(LoaderError::InvalidFormat(format!("Unsupported pickle opcode: 0x{:02x}", opcode)));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

        #[test]

        fn test_unpickler_tuple() {

            let data = b"\x80\x02(K\x01K\x02t.";

            let mut unpickler = PytorchUnpickler::new(data);

            let res = unpickler.parse().unwrap();

            match res {

                PickleValue::Tuple(items) => {

                    assert_eq!(items.len(), 2);

                    match (&items[0], &items[1]) {

                        (PickleValue::Int(1), PickleValue::Int(2)) => {},

                        _ => panic!("Expected Int(1), Int(2), got {:?}", items),

                    }

                }

                _ => panic!("Expected Tuple, got {:?}", res),

            }

        }

    

                #[test]

    

                fn test_unpickler_persid() {

    

                    let data = b"\x80\x02Ptest\n.";

    

                    let mut unpickler = PytorchUnpickler::new(data);

    

                    let res = unpickler.parse().unwrap();

    

                    match res {

    

                        PickleValue::PersistentId(pid) => {

    

                            match *pid {

    

                                PickleValue::String(ref s) => assert_eq!(s, "test"),

    

                                _ => panic!("Expected String('test') as PID, got {:?}", pid),

    

                            }

    

                        }

    

                        _ => panic!("Expected PersistentId, got {:?}", res),

    

                    }

    

                }

    

            

    

                #[test]

    

                fn test_unpickler_binpersid() {

    

                    let data = b"\x80\x02K\x01Q.";

    

                    let mut unpickler = PytorchUnpickler::new(data);

    

                    let res = unpickler.parse().unwrap();

    

                    match res {

    

                        PickleValue::PersistentId(pid) => {

    

                            match *pid {

    

                                PickleValue::Int(1) => {},

    

                                _ => panic!("Expected Int(1) as PID, got {:?}", pid),

    

                            }

    

                        }

    

                        _ => panic!("Expected PersistentId, got {:?}", res),

    

                    }

    

                }

    

            }

    

            

    

        

    
