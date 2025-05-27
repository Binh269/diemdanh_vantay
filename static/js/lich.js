$(document).ready(function () {
    loadTKB();

    function loadTKB() {
        var container = $(".data_tkb");
        container.empty();
        container.append('<div>Đang tải dữ liệu thời khóa biểu...</div>');
        var buttonHtml = `<button class="button_tkb" onclick="them_tkb()">Thêm</button>`;

        $.post('#', { action: 'get_tkb' }, function (data) {
            container.empty();
            container.append(buttonHtml);  
            if (data.length > 0) {
                var tableHtml = `<div class="table-container">
                    <table class="table-tkb">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Ngày</th>
                                <th>Thời Gian</th>
                                <th>Mã Phòng</th>
                                <th>Mã Lớp</th>
                                <th>Mã GV</th>
                                <th>Thao Tác</th>
                            </tr>
                        </thead>
                        <tbody>`;
                data.forEach(function (tkb) {
                    tableHtml += `
                    <tr class="tkb-row" data-id="${tkb.id}">
                        <td>${tkb.id}</td>
                        <td>${tkb.date}</td>
                        <td>${tkb.time}</td>
                        <td>${tkb.maphong}</td>
                        <td>${tkb.malop}</td>
                        <td>${tkb.magv}</td>
                        <td>
                            <button class="button_tkb" onclick="sua_tkb('${tkb.id}', this)">Sửa</button>
                            <button class="button_tkb" onclick="xoa_tkb('${tkb.id}', this)">Xóa</button>
                        </td>
                    </tr>`;
                });
                tableHtml += `</tbody></table></div>`;
                container.append(tableHtml);
            } else {
                container.append('<div>Không có dữ liệu thời khóa biểu.</div>');
            }
        }, 'json');
    }
    window.loadtimkiemtkb = function () {
        var timkiem = $("#timkieminput").val().trim();
        var container = $(".data_tkb");
        container.empty();
        container.append('<div>Đang tải dữ liệu thời khóa biểu...</div>');
        var buttonHtml = `<button class="button_tkb" onclick="them_tkb()">Thêm</button>`;

        $.post('#', { action: 'timkiem_tkb', timkiem: timkiem }, function (data) {
            container.empty();
            container.append(buttonHtml);
            if (data.length > 0) {
                var tableHtml = `<div class="table-container">
                    <table class="table-tkb">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Ngày</th>
                                <th>Thời Gian</th>
                                <th>Mã Phòng</th>
                                <th>Mã Lớp</th>
                                <th>Mã GV</th>
                                <th>Thao Tác</th>
                            </tr>
                        </thead>
                        <tbody>`;
                data.forEach(function (tkb) {
                    tableHtml += `
                    <tr class="tkb-row" data-id="${tkb.id}">
                        <td>${tkb.id}</td>
                        <td>${tkb.date}</td>
                        <td>${tkb.time}</td>
                        <td>${tkb.maphong}</td>
                        <td>${tkb.malop}</td>
                        <td>${tkb.magv}</td>
                        <td>
                            <button class="button_tkb" onclick="sua_tkb('${tkb.id}', this)">Sửa</button>
                            <button class="button_tkb" onclick="xoa_tkb('${tkb.id}', this)">Xóa</button>
                        </td>
                    </tr>`;
                });
                tableHtml += `</tbody></table></div>`;
                container.append(tableHtml);
            } else {
                container.append('<div>Không có dữ liệu thời khóa biểu.</div>');
            }
        }, 'json');
    }

    $("#timkiemtkb").on("click", function () {
        loadtimkiemtkb();
    });
    window.them_tkb = function () {
        var html = `
        <form id="addTKBForm">
            <div class="mb-3">
                <label for="idInput" class="form-label">ID</label>
                <input type="text" class="form-control" id="idInput" placeholder="Nhập ID">
            </div>
            <div class="mb-3">
                <label for="dateInput" class="form-label">Ngày</label>
                <input type="date" class="form-control" id="dateInput" placeholder="Nhập ngày">
            </div>
            <div class="mb-3">
                <label for="timeInput" class="form-label">Thời Gian</label>
                <input type="time" class="form-control" id="timeInput" placeholder="Nhập thời gian">
            </div>
            <div class="mb-3">
                <label for="maphongInput" class="form-label">Mã Phòng</label>
                <input type="text" class="form-control" id="maphongInput" placeholder="Nhập mã phòng">
            </div>
            <div class="mb-3">
                <label for="magvInput" class="form-label">Mã GV</label>
                <input type="text" class="form-control" id="magvInput" placeholder="Nhập mã giáo viên">
            </div>
            <div class="mb-3">
                <label for="malopInput" class="form-label">Mã Lớp</label>
                <input type="text" class="form-control" id="malopInput" placeholder="Nhập mã lớp">
            </div>
        </form>`;

        var confirmBox = $.confirm({
            title: 'Thêm Thời Khóa Biểu',
            content: html,
            boxWidth: '30%',
            theme: 'material',
            buttons: {
                add: {
                    text: 'Thêm',
                    btnClass: 'btn-primary',
                    action: function () {
                        var id = $('#idInput').val();
                        var date = $('#dateInput').val();
                        var time = $('#timeInput').val();
                        var maphong = $('#maphongInput').val();
                        var magv = $('#magvInput').val();
                        var malop = $('#malopInput').val();

                        if (!id || !date || !time || !maphong || !magv || !malop) {
                            $.alert('Vui lòng nhập đầy đủ thông tin!');
                            return false;
                        }

                        $.post('#', {
                            action: 'add_tkb',
                            id: id,
                            date: date,
                            time: time,
                            maphong: maphong,
                            magv: magv,
                            malop: malop
                        }, function (data) {
                            if (data.ok) {
                                $.alert('Đã Thêm Thành Công');
                                loadTKB(); 
                                confirmBox.close();
                            } else {
                                $.alert('Thêm Thất Bại');
                            }
                        }, 'json');
                        return false;
                    }
                },
                cancel: {
                    text: 'Hủy',
                    btnClass: 'btn-secondary',
                }
            }
        });
    };


    window.sua_tkb = function (id, button) {
        var row = $(button).closest('.tkb-row');
        var currentDate = row.find('td').eq(1).text().trim();
        var currentTime = row.find('td').eq(2).text().trim();
        var currentMaphong = row.find('td').eq(3).text().trim();
        var currentMalop = row.find('td').eq(4).text().trim();
        var currentMagv = row.find('td').eq(5).text().trim();

        var html = `
        <form id="editTKBForm">
            <div class="mb-3">
                <label for="dateEditInput" class="form-label">Ngày</label>
                <input type="date" class="form-control" id="dateEditInput" value="${currentDate}">
            </div>
            <div class="mb-3">
                <label for="timeEditInput" class="form-label">Thời Gian</label>
                <input type="time" class="form-control" id="timeEditInput" value="${currentTime}">
            </div>
            <div class="mb-3">
                <label for="maphongEditInput" class="form-label">Mã Phòng</label>
                <input type="text" class="form-control" id="maphongEditInput" value="${currentMaphong}">
            </div>
            <div class="mb-3">
                <label for="malopEditInput" class="form-label">Mã Lớp</label>
                <input type="text" class="form-control" id="malopEditInput" value="${currentMalop}">
            </div>
            <div class="mb-3">
                <label for="magvEditInput" class="form-label">Mã GV</label>
                <input type="text" class="form-control" id="magvEditInput" value="${currentMagv}">
            </div>
        </form>
        `;

        $.confirm({
            title: 'Chỉnh Sửa Thời Khóa Biểu',
            content: html,
            boxWidth: '30%',
            useBootstrap: false,
            buttons: {
                save: {
                    text: 'Lưu',
                    btnClass: 'btn-primary',
                    action: function () {
                        var newDate = $('#dateEditInput').val();
                        var newTime = $('#timeEditInput').val();
                        var newMaphong = $('#maphongEditInput').val();
                        var newMalop = $('#malopEditInput').val();
                        var newMagv = $('#magvEditInput').val();

                        $.post('#', {
                            action: 'update_tkb',
                            id: id,
                            date: newDate,
                            time: newTime,
                            maphong: newMaphong,
                            malop: newMalop,
                            magv: newMagv
                        }, function (data) {
                            if (data.ok) {
                                $.alert('Chỉnh Sửa Thành Công');
                                row.find('td').eq(1).text(newDate);
                                row.find('td').eq(2).text(newTime);
                                row.find('td').eq(3).text(newMaphong);
                                row.find('td').eq(4).text(newMalop);
                                row.find('td').eq(5).text(newMagv);
                            } else {
                                $.alert('Chỉnh Sửa Thất Bại');
                            }
                        }, 'json');
                    }
                },
                cancel: {
                    text: 'Hủy',
                    btnClass: 'btn-secondary',
                }
            }
        });
    }

    window.xoa_tkb = function (id, button) {
        if (confirm("Bạn có chắc chắn muốn xóa thời khóa biểu này?")) {
            $.post('#', { action: 'delete_tkb', id: id }, function (response) {
                alert(response.message);
                if (response.ok) {
                    loadTKB();
                }
            }, 'json');
        }
    }

    document.getElementById('exportExcel').addEventListener('click', function () {
        const rows = document.querySelectorAll('.table-tkb tbody .tkb-row');
        const data = [];

        rows.forEach((row) => {
            const cells = row.querySelectorAll('td');
            const rowData = {
                ID: cells[0].textContent.trim(),
                Ngày: cells[1].textContent.trim(),
                'Thời Gian': cells[2].textContent.trim(),
                'Mã Phòng': cells[3].textContent.trim(),
                'Mã Lớp': cells[4].textContent.trim(),
                'Mã GV': cells[5].textContent.trim()
            };
            data.push(rowData);
        });

        const worksheet = XLSX.utils.json_to_sheet(data);
        const workbook = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(workbook, worksheet, 'Thời Khóa Biểu');
        XLSX.writeFile(workbook, 'ThoiKhoaBieu.xlsx');
    });

    let importedData = [];

    document.getElementById('importExcel').addEventListener('change', function (event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();

            reader.onload = function (e) {
                const data = new Uint8Array(e.target.result);
                const workbook = XLSX.read(data, { type: 'array' });
                const firstSheetName = workbook.SheetNames[0];
                const worksheet = workbook.Sheets[firstSheetName];
                importedData = XLSX.utils.sheet_to_json(worksheet);

                const tbody = document.querySelector('.table-tkb tbody');
                tbody.innerHTML = '';

                importedData.forEach((row, index) => {
                    const tr = document.createElement('tr');
                    tr.className = 'tkb-row';
                    tr.setAttribute('data-id', row.id);

                    tr.innerHTML = `
                    <td>${row.id || ''}</td>
                    <td>${row['date'] || ''}</td>
                    <td>${row['time'] || ''}</td>
                    <td>${row['maphong'] || ''}</td>
                    <td>${row['malop'] || ''}</td>
                    <td>${row['magv'] || ''}</td>
                    <td>Chưa Lưu</td>
                `;
                    tbody.appendChild(tr);
                });

                document.getElementById('saveToDB').disabled = false; 
            };

            reader.readAsArrayBuffer(file);
        }
    });
    document.getElementById('saveToDB').addEventListener('click', function () {
        if (importedData.length === 0) {
            alert('Không có dữ liệu để lưu!');
            return;
        }

        $.post('#', {
            action: 'import_tkb',
            data: JSON.stringify(importedData) 
        }, function (response) {
            if (response.ok) {
                alert('Dữ liệu đã được lưu thành công!');
                loadTKB(); 
                importedData = [];
                document.getElementById('saveToDB').disabled = true; 
            } else {
                alert('Lưu dữ liệu thất bại: ' + response.message);
            }
        }, 'json');
    });
    document.getElementById('saveToDB').addEventListener('click', function () {
        if (importedData.length === 0) {
            alert('Không có dữ liệu để lưu!');
            return;
        }

        $.post('#', {
            action: 'import_tkb',
            data: JSON.stringify(importedData) 
        }, function (response) {
            if (response.ok) {
                alert('Dữ liệu đã được lưu thành công!');
                loadTKB(); 
                importedData = [];
                document.getElementById('saveToDB').disabled = true; 
            } else {
                alert('Lưu dữ liệu thất bại: ' + response.message);
            }
        }, 'json');
    });

});
